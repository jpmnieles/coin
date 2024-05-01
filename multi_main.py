import argparse
import getpass

import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
from torchvision import transforms
from torchvision.utils import save_image
from collections import OrderedDict

import rff
from PIL import Image
from tqdm import tqdm
import imageio
import random

from torchngp import functional
import util


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ne", "--num_epochs", help="Number of iterations to train for", type=int, default=10000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=1e-2)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=143)
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-hd", "--hidden_dim", help="Layer sizes as list of ints", type=int, default=32)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=4)
parser.add_argument("-bs", "--batch_size", help="w0 parameter for SIREN model.", type=int, default=2**17)
parser.add_argument("-en", "--encoding", help="w0 parameter for first layer of SIREN model.", type=str, default="gaussian")
input_dim = 2

args = parser.parse_args()


# Helper Functions
@torch.no_grad()
def render_image(net, ncoords, image_shape, mean, std, batch_size, input_dim) -> torch.Tensor:
    C, H, W = image_shape
    parts = []
    for batch in ncoords.reshape(-1, input_dim).split(batch_size):
        parts.append(net(batch))
    img = torch.cat(parts, 0)
    img = img.view((H, W, C)).permute(2, 0, 1)
    img = torch.clip(img * std + mean, 0, 1)
    return img

class Trainer():
    def __init__(self, representation, batch_size, num_epochs, lr=1e-3, print_freq=1):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

    def train(self, encoords, nimg, num_steps):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """                           
        # Train
        pbar = tqdm(range(num_steps), mininterval=0.1)
        pbar_postfix = {"loss": 0.0}
        encoords_flat = encoords.reshape(-1, encoords.shape[2])
        nimg_flat = nimg.view(nimg.shape[0], -1)
        for epoch in range(self.num_epochs):
            ids = torch.randperm(encoords_flat.shape[0])
            for batch in ids.split(self.batch_size):
                batch_coordinates = encoords_flat[batch]
                batch_features = nimg_flat[:, batch]

                batch_predicted = self.representation(batch_coordinates)
                self.optimizer.zero_grad()
                loss = self.loss_func(batch_predicted, batch_features.permute(1, 0))
                loss.backward()
                self.optimizer.step()

                # Calculate psnr
                pbar_postfix["loss"] = loss.item()
                pbar.set_postfix(**pbar_postfix, refresh=False)
                pbar.update()

            recimg = render_image(self.representation, encoords, nimg.shape, mean, std, self.batch_size, encoords.shape[2])
            psnr, _ = functional.peak_signal_noise_ratio(
                img.unsqueeze(0), recimg.unsqueeze(0), 1.0
            )
            psnr = psnr.mean().item()
            pbar_postfix["psnr[dB]"] = psnr
            pbar_postfix["best psnr[dB]"] = self.best_vals['psnr']
            
            if psnr > self.best_vals['psnr']:
                self.best_vals['psnr'] = psnr
                # If model achieves best PSNR seen during training, update
                # model
                if epoch > int(self.num_epochs / 2.):
                    for k, v in self.representation.state_dict().items():
                        self.best_model[k].copy_(v)


# Network
class CompressionModule(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        
        # Input layer
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        
        x = self.output_layer(x)
        return x


# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.full_dataset:
    min_id, max_id = 1, 24  # Kodak dataset runs from kodim01.png to kodim24.png
else:
    min_id, max_id = args.image_id, args.image_id

# Dictionary to register mean values (both full precision and half precision)
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)



# Fit images
for i in range(min_id, max_id + 1):
    print(f'Image {i}')

    # Load image
    img = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
    
    # Read image
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    img = img.to(device)
    coords = functional.make_grid(img.shape[1:], indexing="xy", device=device)
    ncoords = functional.normalize_uv(coords, coords.shape[:-1], indexing="xy")

    # Compute image stats
    mean, std = img.mean((1, 2), keepdim=True), img.std((1, 2), keepdim=True)
    n_pixels = np.prod(img.shape[1:])

    # Normalize image
    nimg = (img - mean) / std

    # Encoding
    if args.encoding == 'positional':
        encoder = rff.layers.PositionalEncoding(sigma=0.5, m=10)
    elif args.encoding == 'gaussian':
        encoder = rff.layers.GaussianEncoding(sigma=2.0, input_size=2, encoded_size=64)
    else:
        encoder = lambda x: x

    encoords = encoder(ncoords)
    encoords = encoords.to(device, dtype)

    # Setup model
    model = CompressionModule(
        input_dim=encoords.shape[2],
        output_dim=3,
        num_hidden_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
    )
    model.to(device)

    # Estimation of total steps such that each pixel is selected num_epochs times
    n_pixels = np.prod(img.shape[1:])
    n_steps_per_epoch = max(n_pixels // args.batch_size, 1)
    n_steps = int(args.num_epochs * n_steps_per_epoch)

    # Set up training
    trainer = Trainer(model, args.batch_size, args.num_epochs, lr=args.learning_rate)
    model_size = util.model_size_in_bits(model) / 8000
    print(f'Model size: {model_size:.1f}kB')
    fp_bpp = util.bpp(model=model, image=img)
    print(f'Half precision bpp: {fp_bpp/2:.2f}')
    trainer.train(encoords, nimg, n_steps)

    # Full Precision
    model_size = util.model_size_in_bits(model) / 8000.
    print(f'Model size: {model_size:.1f}kB')
    fp_bpp = util.bpp(model=model, image=img)
    print(f'Full precision bpp: {fp_bpp:.2f}')
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    # Log full pre2ci2sion results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Display
    imgrec = render_image(model, encoords, nimg.shape, mean, std, args.batch_size, encoords.shape[2])
    err = (imgrec - img).abs().sum(0)

    # Saving image reconstruction
    with torch.no_grad():
        save_image(imgrec.to('cpu'), args.logdir + f'/fp_reconstruction_{i}.png')

    # Save best model
    torch.save(trainer.best_model, args.logdir + f'/best_model_{i}.pt')

    # Half Precision
    model.load_state_dict(trainer.best_model)

    half_model = model.half().to('cuda')
    half_encoords = encoords.half().to('cuda')

    model_size = util.model_size_in_bits(half_model) / 8000.
    hp_bpp = util.bpp(model=half_model, image=img)
    half_imgrec = render_image(half_model, half_encoords, nimg.shape, mean, std, args.batch_size, encoords.shape[2])
    half_err = (half_imgrec - img).abs().sum(0)
    half_psnr, _ = functional.peak_signal_noise_ratio(
        img.unsqueeze(0), half_imgrec.unsqueeze(0), 1.0
    )

    # Log full precision results
    print(f'Model size: {model_size:.1f}kB')
    print(f'Half precision bpp: {hp_bpp:.2f}')
    print(f'PSNR: {half_psnr.mean().item():.2f}')
    results['hp_bpp'].append(hp_bpp)
    results['hp_psnr'].append(half_psnr.mean().item())

    # Saving image reconstruction
    with torch.no_grad():
        save_image(half_imgrec.to('cpu'), args.logdir + f'/hp_reconstruction_{i}.png')

    # Save logs for individual image
    with open(args.logdir + f'/logs{i}.json', 'w') as f:
        json.dump(trainer.logs, f)

    print('\n')

print('Full results:')
print(results)
with open(args.logdir + f'/results.json', 'w') as f:
    json.dump(results, f)

# Compute and save aggregated results
results_mean = {key: util.mean(results[key]) for key in results}
with open(args.logdir + f'/results_mean.json', 'w') as f:
    json.dump(results_mean, f)

print('Aggregate results:')
print(f'Full precision, bpp: {results_mean["fp_bpp"]:.2f}, psnr: {results_mean["fp_psnr"]:.2f}')
print(f'Half precision, bpp: {results_mean["hp_bpp"]:.2f}, psnr: {results_mean["hp_psnr"]:.2f}')
