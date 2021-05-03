from torch.nn import (
    Module, ModuleList, LeakyReLU
) 

from torch import randn, device
from collections import OrderedDict

from custom import (
    GeneratorInitialBlock, 
    GeneratorBlock, 
    DiscriminatorInitialBlock,
    DiscriminatorMidBlock,
    DiscriminatorFinalBlock
    )


class Generator(Module):
    "Generator of the MSG-GAN."

    def __init__(self, 
            num_blocks=2,
            channels=[512,512,512,512,512,256,128,64,32,16],
            kernel_size=3,
            padding=1,
            stride=1,
            initial_spatial_dim=4, 
            img_channels=3,
            activation=LeakyReLU(0.2),
            scale_factor=2,
            ):
        super().__init__()
        
        err = "num blocks must be less then len(channels)."
        assert num_blocks < len(channels), err
        
        self.blocks = ModuleList()
        self.num_blocks= num_blocks
        for blk in range(num_blocks):
            self.blocks.append(
                GeneratorInitialBlock(
                    in_dimension=channels[blk],
                    spatial_dimension=initial_spatial_dim,
                    out_channels=channels[blk+1],
                    img_channels=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation
                ) if blk == 0 else
                GeneratorBlock(
                    in_channels=channels[blk],
                    out_channels=channels[blk+1],
                    scale_factor=scale_factor,
                    img_channels=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation
                )
            )
    
    def forward(self, x, generate_images=range(9)):
        imgs_out = OrderedDict()

        for blk in range(self.num_blocks):
            x, imgs_out[blk] = (
                self.blocks[blk](x, generate_img=blk in generate_images)
            )

        return imgs_out
    

class Discriminator(Module):
    "Discriminator of the MSG-GAN."

    def __init__(
            self, 
            num_blocks=2,
            channels=[16, 32, 64, 128, 256, 512, 512, 512, 512, 512],
            kernel_size=3,
            padding=1,
            stride=1,
            final_spatial_dim=4, 
            img_channels=3,
            activation=LeakyReLU(0.2),
            dimension_reduction=2,
            scheme="simple"
            ):
        super().__init__()

        err = "num blocks must be less then len(channels)."
        assert num_blocks < len(channels), err

        self.blocks = ModuleList()
        self.num_blocks= num_blocks    
        for blk in range(num_blocks-1):
            idx = blk+len(channels)-num_blocks-1
            self.blocks.append(
                DiscriminatorInitialBlock(
                    in_channel=channels[idx],
                    out_channel=channels[idx+1],
                    img_channel=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                    dimension_reduction=dimension_reduction
                ) if blk == 0 else 
                DiscriminatorMidBlock(
                    in_channel=channels[idx],
                    out_channel=channels[idx+1],
                    img_channel=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                    dimension_reduction=dimension_reduction,
                    scheme=scheme                    
                )
            )
        idx = len(channels)-2
        self.blocks.append(
            DiscriminatorFinalBlock(
                    in_channel=channels[idx],
                    out_channel=channels[idx+1],
                    img_channel=img_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                    spatial_dimension=final_spatial_dim,
                    scheme=scheme   
            )
        )

    def forward(self, img_dict):

        idx = list(img_dict.keys())[::-1]
        x = self.blocks[0](img_dict[idx[0]])
        for blk in range(1, self.num_blocks):
            x = self.blocks[blk](x, img_dict[idx[blk]])
        return x
    

if __name__ == "__main__":
    
    dev = device("cuda:0")
    for i in range(100):
        z = randn(5, 512, 1).to(dev)
        gen = Generator(6).to(dev)
        dis = Discriminator(6).to(dev)
        h = gen(z)
        print(dis(h).shape)


