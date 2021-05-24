from torch import Tensor, rand, randn
from numpy import ones
from model import Discriminator

def gradient_penalty_loss(discriminator, from_real, from_fake):

    epsilon = rand(
        size=(from_real.shape[0], *ones(len(from_real.shape) - 1)),
        device=from_real.device(), requires_grad=True
    )
    x_hat = (epsilon * from_real + (1-epsilon) * from_fake).require_grad_(True)
    D_x_hat = Discriminator(x_hat)
    print(D_x_hat.shape)



    





