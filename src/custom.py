from torch.nn import (
    Module, Linear, Conv2d, Upsample, 
    AvgPool2d, LeakyReLU, Sequential
    )

from torch import (
    sqrt, var, cat, randn, Tensor, device,
    squeeze
)

import torch 

from time import time

def pixel_norm(x, epsilon=1e-8):
    """Return the pixel norm of the input.
    Input: activation of size NxCxWxH 
    """
    return x / sqrt((x**2.).mean(axis=1, keepdim=True) + epsilon)


def minbatchstd(x, group_size=5):
    """Implementation of the minbatch standard deviation.
    """
    
    x_size = x.shape
    err = "Batch size must be divisible by group size"
    assert x_size[0] % group_size == 0, err
    
    group_len = x_size[0] // group_size
    y = x.view(group_len, group_size, *x_size[1:])
    y = var(y, dim=1)
    y = y.view(group_len, -1)
    y = y.mean(dim=1).view(group_len, 1)
    y = y.expand(group_len, x_size[2] * x_size[3])
    y = y.view(group_len, 1, 1, x_size[2], x_size[3])
    y = y.expand(-1, group_size, -1, -1, -1)
    y = y.reshape(-1, 1, x_size[2], x_size[3])
    x = cat([x, y], dim=1)
    return x


class PhiScheme(Module):
    "Defines the phi scheme used in the paper."

    def __init__(self, img_channels, in_channels, scheme="simple") -> Tensor:
        super().__init__()

        self.scheme = scheme
        schemes = ["simple", "lin_cat", "cat_lin"]
        err = "Select one of {} {} {} as scheme.".format(*schemes)
        assert scheme.lower() in schemes, err

        self.r_prime = Conv2d(
                in_channels=img_channels, 
                out_channels=in_channels, 
                kernel_size=1) if scheme in schemes[1:] else None
        
    def forward(self, x1, x2):
        if self.scheme == "simple":
            return cat([x1,x2], dim=1)
        elif self.scheme == "lin_cat":
            return cat([self.r_prime(x1), x2], dim=1)
        elif self.scheme == "cat_lin":
            return self.r_prime(cat([x1, x2], dim=1))
        else:
            raise("No valid scheme is selected")
        

class FromRGB(Module):
    "Implementation of FromRGB 0 in paper [1] page 11"

    def __init__(self, img_channels, out_channels) -> Tensor:
        super().__init__()

        self.from_rgb = Conv2d(
            in_channels=img_channels,
            out_channels=out_channels,
            kernel_size=1
        ) 
    
    def forward(self, x):
        return self.from_rgb(x)


class GeneratorInitialBlock(Module):
    """Implements the initial block of the generator.
    """

    def __init__(
            self, in_dimension=512, spatial_dimension=4, out_channels=512, 
            img_channels=3, kernel_size = (3,3), stride = (1,1), 
            padding = (1,1), activation=LeakyReLU(0.2)) -> Tensor:
        super().__init__()
        
        self.in_dimension = in_dimension
        self.spatial_dimension = spatial_dimension
        self.out_channels = out_channels
        self.img_channels = img_channels

        self.activation = activation
        self.linear = Linear(1, int(spatial_dimension ** 2.))
        self.conv = Conv2d(
            in_channels=in_dimension, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding)
        self.img_conv = Conv2d(
            in_channels=self.out_channels, 
            out_channels=self.img_channels, 
            kernel_size=1)
        
    def forward(self, x, generate_img=True):
        x = self.linear(x)
        x = x.view(-1, self.in_dimension, 
            self.spatial_dimension, self.spatial_dimension)
        x = pixel_norm(self.activation(x))
        x = pixel_norm(self.activation(self.conv(x)))
        if generate_img:
            return x, self.img_conv(x)
        else:
            return x, None

    def extra_repr(self) -> str:
        return "in dim {0}, out_dim {1}x{2}x{2}".format(
            self.in_dimension, self.out_channels, self.spatial_dimension
        )


class GeneratorBlock(Module):
    "Generator mid blocks."

    def __init__(
            self, in_channels=512, out_channels=512, 
            scale_factor=2, img_channels=3, kernel_size=(3,3), 
            stride=(1,1), padding=(1,1), 
            activation=LeakyReLU(0.2)) -> Tensor:
        super().__init__()

        self.activation = activation
        self.upsample = Upsample(scale_factor=scale_factor)
        self.conv_first = Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.conv_second = Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.img_conv = Conv2d(
            in_channels=out_channels, 
            out_channels=img_channels, 
            kernel_size=1)        

    def forward(self, x, generate_img=True):
        x = self.upsample(x)
        x = pixel_norm(self.activation(self.conv_first(x)))
        x = pixel_norm(self.activation(self.conv_second(x)))
        if generate_img:
            return x, self.img_conv(x)
        else:
            return x, None        


class DiscriminatorFinalBlock(Module):
    "Implementation of the final block of the discriminator/critic"

    def __init__(
            self, in_channel, spatial_dimension=4,
            img_channel=3, kernel_size = (3,3), stride = (1,1), 
            padding = (1,1), activation=LeakyReLU(0.2), 
            scheme="simple") -> Tensor:
        super().__init__()

        self.activation = activation
        self.concat = PhiScheme(
            img_channels=img_channel, 
            in_channels=in_channel, 
            scheme=scheme
            )
        self.conv_first = Conv2d(
            in_channels=img_channel + in_channel + 1,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        self.conv_second = Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=4
            )
        self.linear = Linear(
            in_features=in_channel,
            out_features=1)
    
    def forward(self, a_prime, o):
        x = minbatchstd(self.concat(a_prime, o))
        x = self.activation(self.conv_first(x))
        x = self.activation(self.conv_second(x))
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        return self.linear(x)


class DiscriminatorMidBlock(Module):
    "Implementation of the final block of the discriminator/critic"

    def __init__(
            self, in_channel, out_channel, img_channel=3, 
            dimension_reduction=2, kernel_size=(3,3), stride=(1,1), 
            padding = (1,1), activation=LeakyReLU(0.2), 
            scheme="simple") -> Tensor:
        super().__init__()

        self.activation = activation
        self.concat = PhiScheme(
            img_channels=img_channel, 
            in_channels=in_channel, 
            scheme=scheme
            )
        self.conv_first = Conv2d(
            in_channels=img_channel + in_channel + 1,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        self.conv_second = Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        self.avg_pool = AvgPool2d(
            kernel_size=dimension_reduction, 
            stride=dimension_reduction
            )
    
    def forward(self, a_prime, o):
        x = minbatchstd(self.concat(a_prime, o))
        x = self.activation(self.conv_first(x))
        x = self.activation(self.conv_second(x))
        return self.avg_pool(x)



        
        








if __name__ == "__main__":
    
    dev = device("cuda:0")
    a_prime = randn(5, 512, 8, 8).to(dev)
    o1 = randn(5, 3, 8, 8).to(dev)
    o2 = randn(5, 3, 4, 4).to(dev)
    d_m = DiscriminatorMidBlock(512, 512).to(dev)
    d = DiscriminatorFinalBlock(512).to(dev)
    print(d(d_m(a_prime, o1), o2).shape)

"""     a1 = GeneratorInitialBlock(512, 4, 512).to(dev)
    a2 = GeneratorBlock(512,512,2).to(dev)
    a3 = GeneratorBlock(512,512,2).to(dev) 
    a4 = GeneratorBlock(512,512,2).to(dev) 
    a5 = GeneratorBlock(512,256,2).to(dev) 
    a6 = GeneratorBlock(256,128,2).to(dev) 

    for i in range(10000):
        
        y = a6(a5(a4(a3(a2(a1(x)[0])[0])[0])[0])[0])[0]
        print(y.shape)
 """
    