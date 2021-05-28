from torch.optim import RMSprop
import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor
from os import getcwd
from time import time, sleep

from .src.loss import WGANGP_loss
from .src.model import Generator, Discriminator
from .src.custom import DataLoader, LossTracker



# Parse the arguments of the model
parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.003, help="RMSPROP: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent space")
parser.add_argument("--num_blocks_gen", type=int, default=6, help="number of generator blocks")
parser.add_argument("--num_blocks_dis", type=int, default=6, help="number of discriminator blocks")
parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU for training")
parser.add_argument("--save_skips", type=int, default=20, help="number of epochs to skip saving the model")
parser.add_argument("--continue_checkpoint", type=bool, default=True, help="Continue from the last checkpoint")
parser.add_argument("--save_dir", type=str, default=f"{getcwd()}/weights/", help="Continue from the last checkpoint")
parser.add_argument("--images", type=str, default=f"{getcwd()}/images/", help="Continue from the last checkpoint")
parser.add_argument("--n_critic", type=int, default=1, help="the number of critic iterations per generator iteration")
parser.add_argument("--dataset", type=str, default="CIFAR10", help="Selected dataset either CIFAR10 or CelebA")
parser.add_argument("--conv_crit", type=float, default=1.e-4, help="Convergence Criterion")



opt = parser.parse_args()

# Check if the required directories exist
if not os.Path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)
if not os.Path.exists(opt.images):
    os.mkdir(opt.images)

device = torch.device("cuda:0") if opt.use_gpu else torch.device("cpu") 

# create the models
generator = Generator(opt.num_blocks_gen).to(device)
discriminator = Discriminator(opt.num_blocks_dis).to(device)

# Load the models if necessary
if opt.continue_checkpoint:
    generator = torch.load(f'{opt.save_dir}/generator.to')
    discriminator = torch.load(f'{opt.save_dir}/discriminator.to')

# Define the optimizers
generator_optimizer = RMSprop(
    generator.parameters, 
    lr=opt.lr, 
    betas=(opt.b1, opt.b2)
    ).to(device)
discriminator_optimizer = RMSprop(
    generator.parameters, 
    lr=opt.lr, 
    betas=(opt.b1, opt.b2)
    ).to(device)

# loading the data
data_loader = DataLoader(
    opt.dataset, opt.batch_size, opt.num_blocks_dis
    )

# track loss stats of the model
loss_tracker = LossTracker(100, 100)

# while theta has not converged do: 
# from Improved training of Wasserstein GANs p-4

converged = False
loss_tracker = []
while not converged:
    
    start_time = time()
    loss_tracker = []
    for t in range(opt.n_disc):
        latent_variable = torch.randn(
            (opt.batch_size, 512), requires_grad=True
            ).to(device)

        fake_results = generator(latent_variable)
        real_results = data_loader.load_images()

        discriminator_optimizer.zero_grad()
        discriminator_loss = WGANGP_loss(
            discriminator=discriminator, 
            from_real=real_results,
            from_fake=fake_results
            )
        loss_tracker.append(discriminator_loss.tolist())
        discriminator_loss.backward()
        discriminator_optimizer.step()
    
    










