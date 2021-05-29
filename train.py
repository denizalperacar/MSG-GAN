from torch.optim import RMSprop
import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.transforms import Normalize
from torchvision.utils import save_image
from os import getcwd
from time import time, sleep
import numpy as np
from src.loss import WGANGP_loss
from src.model import Generator, Discriminator
from src.custom import LossTracker, get_latent_variable
from src.data_loader import DataLoader


# Parse the arguments of the model
parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.003, help="RMSPROP: learning rate")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--num_blocks", type=int, default=4, help="number of generator blocks")
parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU for training")
parser.add_argument("--save_skips", type=int, default=200, help="number of epochs to skip saving the model")
parser.add_argument("--continue_checkpoint", type=bool, default=False, help="Continue from the last checkpoint")
parser.add_argument("--save_dir", type=str, default=f"{getcwd()}/weights/", help="Continue from the last checkpoint")
parser.add_argument("--images", type=str, default=f"{getcwd()}/images/", help="Continue from the last checkpoint")
parser.add_argument("--n_disc", type=int, default=1, help="the number of discriminator iterations per generator iteration")
parser.add_argument("--dataset", type=str, default="CIFAR10", help="Selected dataset either CIFAR10 or CelebA")
parser.add_argument("--conv_crit", type=float, default=1.e-4, help="Convergence Criterion")
parser.add_argument("--gen_steps", type=int, default=100, help="save the image of the generated images each gen_steps")
parser.add_argument("--lamda", type=float, default=10., help="lamda used in WGAN-GP")

opt = parser.parse_args()

print_string = (
    'iterations {:05d} discriminator loss stats' 
    + '|mean {:16.08f} |min {:16.08f} | max {:16.08f} '
    + '| std {:16.08f} |time {:16.08f}s'
    )

# Check if the required directories exist
if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)
if not os.path.exists(opt.images):
    os.mkdir(opt.images)

device = torch.device("cuda:0") if opt.use_gpu else torch.device("cpu") 

# create the models
generator = Generator(
    opt.num_blocks,
    channels=np.array([512,512,512,512,512,256,128,64,32,16]).astype(int)//4
    ).to(device)
discriminator = Discriminator(
    opt.num_blocks,
    channels=np.array([16, 32, 64, 128, 256, 512, 512, 512, 512, 512]).astype(int)//4
    ).to(device)

# Load the models if necessary
if opt.continue_checkpoint:
    generator = torch.load(f'{opt.save_dir}/generator')
    discriminator = torch.load(f'{opt.save_dir}/discriminator')

# Define the optimizers
generator_optimizer = RMSprop(
    generator.parameters(), 
    lr=opt.lr, 
    )
discriminator_optimizer = RMSprop(
    discriminator.parameters(), 
    lr=opt.lr, 
    )

# loading the data
data_loader = DataLoader(
    opt.dataset, opt.batch_size, opt.num_blocks, device
    )

# track loss stats of the model
discriminator_loss_tracker = LossTracker(
    100, 100, save_name="disc", eps=opt.conv_crit
    )
generator_loss_tracker = LossTracker(
    100, 100, save_name="gen", eps=opt.conv_crit
    )

# while theta has not converged do: 
# from Improved training of Wasserstein GANs p-4

converged = False
loss_tracker = []
counter = 0
start_time = time()
while not converged:
    
    counter += 1
    loss_tracker = []
    for t in range(opt.n_disc):
        latent_variable = get_latent_variable(
            opt.batch_size, opt.latent_dim, device
            )

        fake_images = generator(latent_variable)
        real_images = data_loader.load_images()

        discriminator_optimizer.zero_grad()
        discriminator_loss = WGANGP_loss(
            discriminator=discriminator, 
            from_real=real_images,
            from_fake=fake_images,
            lamda=opt.lamda
            )
        discriminator_loss_tracker.append(discriminator_loss.tolist())
        discriminator_loss.backward()
        discriminator_optimizer.step()
    
    # update the generator
    latent_variable = get_latent_variable(
        opt.batch_size, opt.latent_dim, device
        )  

    fake_images = generator(latent_variable) 
    generator_optimizer.zero_grad()
    generator_loss = -discriminator(fake_images).mean()
    generator_loss_tracker.append(generator_loss.tolist())
    generator_loss.backward()
    generator_optimizer.step()

    # check the convergence
    gen_conv = generator_loss_tracker.converged()
    dis_conv = discriminator_loss_tracker.converged()
    converged = gen_conv and dis_conv

    if counter % opt.save_skips == 0:
        generator.save(f"{opt.save_dir}/generator")
        discriminator.save(f"{opt.save_dir}/discriminator")
    
    if counter % opt.gen_steps == 0:
        

        latent_variable = get_latent_variable(
            20, opt.latent_dim, device
            )
        fake_images = generator(latent_variable)[opt.num_blocks-1].to("cpu")
        save_image(fake_images.data[:20], f"{opt.images}/%d.png" % counter, nrow=5, value_range=(-1,1), normalize=True)
        
        # statistics of discriminator
        print(print_string.format(
            counter, 
            *discriminator_loss_tracker.loss_tracker[-1],
            time()-start_time))
        start_time = time()














