"""This module contains an implementation of a multi-level hash encoding module.

See:
https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn
import torch.nn.functional as F
from typing import Tuple

from .. import functional


@dataclass
class LevelInfo:
    res: int
    shape: Tuple[int, ...]
    dense: bool
    hashing: bool
    n_encodings: int
    fn_forward: Callable[[torch.Tensor, int], torch.Tensor]
    fn_hash: Callable[[torch.Tensor, Tuple[int, ...], int], torch.LongTensor]


class MultiLevelHybridHashEncoding(torch.nn.Module):
    """Implementation of a multi-level hash encoding.

    Maps spatial coordinates to feature vectors using bilinear
    interpolation of encoding vectors assigned to nodes of regular
    grids at increasing resolution.

    The encoding takes spatial (2D/3D) coordinates and assigns each point
    to an interpolated feature vector per level. Each encoding level `L` is
    defined by a set of nodes arranged in a regular grid (2D/3D) of
    increasing resolution `R`. Each grid node is assigned a specific encoding
    codebook vector, such that each spatial query point can be mapped to `L`
    features of dimension `E` by bilinearly interpolating encoding vectors
    associated with neighboring grid nodes at each level.

    Each level `L` has a maximum number of encoding vectors `T`, such that when
    `R**input_dims > T` a one-to-one mapping cannot exist and codebook vectors
    are shared by multiple grid nodes. The mapping is then realized by a
    hash function that, roughly speaking, ensures that all codebook vectors
    are uniformly assigned to any possible sub-volume of the grid.

    In this pure PyTorch implementation, the encoding is realized by a
    hybrid strategy. For levels of low resolution, vertex-to-codebook
    vector indices are precomputed densly for all grid nodes. At query time,
    a dense input map `(E,H,W)` is generated by advanced indexing of
    pre-computed indices and embedding vectors. By treating query points
    as a single-column image of coordinates, we can use
    `torch.nn.functional.grid_sample` to perform the bilinear interpolation.

    While the above approach is fast, for higher resolutions (usually when
    `R**input_dims>256`), it consumes to excessive memory and we resort to a
    sparse approach. That is, instead of pre-computing all grid node-to-encoding,
    we locate the 4/8 supporting grid-nodes for each query point at each level.
    We then compute the associated encoding vectors for each required grid and
    node and perform the bilinear interpolation of associated encoding vectors
    ourself.

    Both strategies are designed to geometrically consider pixels/voxels as
    squares/cubes. The extrema of the normalized query coordinates (-1,1)
    hence refers to the corners of the pixel/voxels and not to their centers.
    This is what `align_corners=False` does in PyTorch.

    See:
    https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
    """

    def __init__(
        self,
        n_encodings: int = 2**14,
        n_input_dims: int = 3,
        n_embed_dims: int = 2,
        n_levels: int = 16,
        min_res: int = 16,
        max_res: int = 512,
        max_n_dense: int = 2 ** (8 * 3),
        init_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        assert n_input_dims in [2, 3], "Only 2D and 3D inputs are supported"

        self.n_levels = n_levels
        self.n_input_dims = n_input_dims
        self.n_embed_dims = n_embed_dims
        self.max_n_encodings = n_encodings
        self.min_res = min_res
        self.max_res = max_res
        self.max_n_dense = max_n_dense

        self.level_infos: list[LevelInfo] = []

        resolutions = _compute_resolutions(
            n_levels=n_levels, min_res=min_res, max_res=max_res
        )

        for level, res in enumerate(resolutions):
            n_elements = res**self.n_input_dims
            is_dense = n_elements <= self.max_n_dense
            is_hashing = n_elements > self.max_n_encodings
            fn_forward = self._forward_dense if is_dense else self._forward_sparse
            fn_hash = _hash_xor if is_hashing else _hash_ravel
            li = LevelInfo(
                res=res,
                shape=(res,) * self.n_input_dims,
                dense=is_dense,
                hashing=is_hashing,
                n_encodings=min(n_elements, self.max_n_encodings),
                fn_forward=fn_forward,
                fn_hash=fn_hash,
            )
            self.level_infos.append(li)

            # Note: the embedding matrices for dense (E,T) and sparse (T,E)
            # levels are permuted. This is done to better match its usage.

            if li.dense:
                dense_ids = self._compute_dense_indices(li)
                self.register_buffer(
                    "level_emb_indices" + str(level),
                    dense_ids,
                )
                self.register_parameter(
                    "level_emb_matrix" + str(level),
                    torch.nn.Parameter(
                        torch.empty(n_embed_dims, li.n_encodings).uniform_(
                            -init_scale, init_scale
                        )
                    ),
                )
            else:
                emb = torch.empty(li.n_encodings, n_embed_dims).uniform_(
                    -init_scale, init_scale
                )
                self.register_parameter(
                    "level_emb_matrix" + str(level),
                    torch.nn.Parameter(emb),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes embedding features for each location and level.

        Params:
            x: (B,2) or (B,3) array of query locations in value range [-1,+1].

        Returns:
            features: (B,L,E) array of features with dims (E)
                for each query (B) and level (L).
        """
        features = []
        for level, li in enumerate(self.level_infos):
            f = li.fn_forward(x, level)
            features.append(f)
        f = torch.stack(features, 1)
        return f

    def _forward_dense(self, x: torch.Tensor, level: int):
        """Returns the multi-resolutional feature emebeddings for all query locations.

        Params:
            x: (B,2) or (B,3) array of query locations in value range [-1,+1].
            level: level index

        Returns:
            features: (B,E) array of features with dims (E)
                for each query (B).
        """
        B, D = x.shape
        # Note, we re-interpret the query locations as a sampling grid by
        # by shuffling the batch dimension into the first image dimension.
        # Turned out to be faster (many times on cpu) than using expand.
        x = x.view(1, B, *([1] * (D - 1)), D)  # (1,B,1,2) or (1,B,1,1,3)
        indices = getattr(self, "level_emb_indices" + str(level))
        embmatrix = getattr(self, "level_emb_matrix" + str(level))
        # Note for myself: don't store the following as a buffer, it won't work.
        # We need to perform the indexing on-line.
        levelmap = embmatrix[:, indices].unsqueeze(0)
        # Bilinearly interpolate the sampling locations using the levelmap.
        f = F.grid_sample(
            levelmap,
            x,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (1,E,B,1) or (1,E,B,1,1)
        # Shuffle back into (B,E) tensor
        return f.view(self.n_embed_dims, B).permute(1, 0)

    def _forward_sparse(self, x: torch.Tensor, level: int):
        embmatrix = getattr(self, "level_emb_matrix" + str(level))
        ids, w = self._compute_sparse_indices(x, level)
        # (B,4,E) or (B,8,E) -> (B,E)
        return (embmatrix[ids] * w[..., None]).sum(1)

    @torch.no_grad()
    def _compute_dense_indices(self, li: LevelInfo) -> torch.LongTensor:
        index_coords = functional.make_grid(li.shape, indexing="xy")
        ids = li.fn_hash(index_coords, li.shape, li.n_encodings)
        return ids

    @torch.no_grad()
    def _compute_sparse_indices(self, x: torch.Tensor, level: int):
        li = self.level_infos[level]

        # Normalized [-1,1] to pixel [-0.5,R+0.5]
        x = (x + 1) * li.res * 0.5 - 0.5  # (B,C)
        c, w, m = _compute_bilinear_params(x, li.shape)

        # Compute indices
        ids = li.fn_hash(c, li.shape, li.n_encodings)

        # Point outside elements to the first element, but set
        # all weights zero to simulate zero-padding.
        w[~m] = 0.0
        ids[~m] = 0
        return ids, w


def _compute_resolutions(
    n_levels: int = 16,
    min_res: int = 16,
    max_res: int = 512,
):
    """Computes grid resolutions for each level

    Equation 2 and 3 in the paper to determine the number of grid vertices
    per resolution level

    https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
    """
    growth_factor = torch.exp(
        (torch.log(torch.tensor(max_res)) - torch.log(torch.tensor(min_res)))
        / (n_levels - 1)
    )
    resolutions = (
        torch.floor(
            torch.tensor(
                [min_res * growth_factor**level for level in range(n_levels)]
            )
        )
        .long()
        .tolist()
    )
    return resolutions


def _compute_bilinear_params(
    x: torch.Tensor, shape: Tuple[int, ...]
) -> Tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    """Computes bilinear/trilinear interpolation parameters

    Params:
        x: (B,C) points with C=2 or C=3
        shape: (H,W) or (D,H,W) of interpolation grid size

    Returns:
        c: (B,4,2) or (B,8,2) corner integer coordinates
        w: (B,4) or (B,8) weights per corner (sum to one)
        m: (B) mask of valid points x.
    """
    B, C = x.shape
    if C == 2:
        c, w, m = _bilinear_params_2d(x, shape)
    elif C == 3:
        c, w, m = _bilinear_params_3d(x, shape)
    else:
        raise NotImplementedError
    return c, w, m


def _bilinear_params_2d(
    x: torch.Tensor, shape: Tuple[int, ...]
) -> Tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    o = x.new_tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
    xl = x.floor()
    xf = x - xl

    # Compute corners
    c = (xl.long().unsqueeze(1) + o[None, ...]).contiguous()

    # Compute mask
    m = ((c >= 0) & (c < c.new_tensor(shape[::-1])[None, None, :])).all(-1)  # B,4

    # Compute weights
    one_min = 1.0 - xf
    w11 = one_min[:, 0] * one_min[:, 1]
    w12 = one_min[:, 0] * xf[:, 1]
    w21 = xf[:, 0] * one_min[:, 1]
    w22 = xf[:, 0] * xf[:, 1]
    w = torch.stack((w11, w12, w21, w22), 1)  # B,4

    return c, w, m


def _bilinear_params_3d(
    x: torch.Tensor, shape: Tuple[int, ...]
) -> Tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    o = x.new_tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.long,
    )

    xl = x.floor()
    xf = x - xl

    # Compute corners
    c = (xl.long().unsqueeze(1) + o[None, ...]).contiguous()

    # Compute mask
    m = ((c >= 0) & (c < c.new_tensor(shape[::-1])[None, None, :])).all(-1)  # B,8

    # Compute weights
    one_min = 1 - xf
    w000 = one_min[:, 0] * one_min[:, 1] * one_min[:, 2]
    w001 = one_min[:, 0] * one_min[:, 1] * xf[:, 2]
    w010 = one_min[:, 0] * xf[:, 1] * one_min[:, 2]
    w011 = one_min[:, 0] * xf[:, 1] * xf[:, 2]
    w100 = xf[:, 0] * one_min[:, 1] * one_min[:, 2]
    w101 = xf[:, 0] * one_min[:, 1] * xf[:, 2]
    w110 = xf[:, 0] * xf[:, 1] * one_min[:, 2]
    w111 = xf[:, 0] * xf[:, 1] * xf[:, 2]

    w = torch.stack((w000, w001, w010, w011, w100, w101, w110, w111), 1)  # B,8

    return c, w, m


def _hash_xor(
    x: torch.LongTensor, shape: Tuple[int, ...], n_encodings: int
) -> torch.LongTensor:
    # See https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
    del shape
    dev = x.device
    dtype = x.dtype
    primes = [1, 2654435761, 805459861]

    ids = torch.zeros(x.shape[:-1], dtype=dtype, device=dev)
    for i in range(x.shape[-1]):
        ids ^= x[..., i] * primes[i]

    return ids % n_encodings


def _hash_ravel(
    x: torch.LongTensor, shape: Tuple[int, ...], n_encodings: int
) -> torch.LongTensor:
    """Computes linear indices from multi-dimensional indices

    Params:
        x: (N,...,d) multi-dimensional indices with dimensions indexed (x,y,z,...,d)
        shape: shape of grid (D,...,H,W)
        n_encodings: max number of encodings. Unused.

    Returns:
        ids: (N,...) flat indices compliant with np.ravel_multi_index with order='F'.
    """
    del n_encodings
    strides = x.new_tensor(shape[::-1]).cumprod(0).roll(1, 0)
    strides[0] = 1
    return (x * strides.expand_as(x)).sum(-1)


__all__ = ["MultiLevelHybridHashEncoding"]
