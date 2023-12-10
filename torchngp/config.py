import torch
from hydra_zen import make_custom_builds_fn
from typing import Tuple, List

build_conf = make_custom_builds_fn(populate_full_signature=True)


def vecs3_to_tensor(v: List[Tuple[float, float, float]]) -> torch.Tensor:
    return torch.tensor(v).float()


Vecs3Conf = build_conf(vecs3_to_tensor)
