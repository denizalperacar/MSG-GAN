import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
from torch.nn.functional import avg_pool2d
from collections import OrderedDict
from torch.utils.data import Dataset
from pickle import load, dump
import torch
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
from glob import glob
import random
import numpy as np

def filter_func(x):
    return True

class ImageDataset(Dataset):

    def __init__(self, image_list:list, transform:transforms):
        super(ImageDataset, self).__init__()
        self.images = image_list
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx:str):
        return self.transform(self.images[idx])

class Images:

    def __init__(
            self, 
            addresses:list[str], 
            class_names:list[str], 
            filter,
            is_pickle:bool=False,
        ):
        self.filter = filter
        self.images = {}
        
        if is_pickle:
            for ind, address in enumerate(addresses):
                class_name = class_names[ind]
                self.images[class_names[ind]] = (
                    self.load_pickle(f"{address}/{class_name}.pickle")
                    )
                class_name = class_names[ind]
                self.__dict__[class_name] = self.images[class_name]
        else:
            for ind, address in enumerate(addresses):
                img_files = glob(f"{address}/*.jpg")
                self.images[class_names[ind]] = []
                for img_file in img_files:
                    img = Image.open(img_file)
                    if self.filter(img):
                        self.images[class_names[ind]].append(img)
                class_name = class_names[ind]
                self.__dict__[class_name] = self.images[class_name]

    def save_pickle(self, address:str):

        for name in self.images.keys():
            with open(f"{address}/{name}.pickle", "wb") as fid:
                dump(self.images[name], fid)
    
    def load_pickle(self, file_dir:str):

        with open(file_dir, "rb") as fid:
            return load(fid)

    def visualize(
            self, name:str,
            is_random:bool=True, 
            figsize=(16, 16), 
            sw = 6, 
            sh = 6
        ):

        n_images = sw * sh
        plt.figure(figsize=figsize)
        choices = range(len(self.images[name]))
        for i in range(n_images):
            if is_random:
                img_id = random.choice(choices)

            plt.subplot(sh, sw, i + 1)
            plt.imshow(np.array(self.images[name][img_id]))
            plt.title(img_id)
            plt.xticks([])
            plt.yticks([])

        plt.show()

    def get_dataset(
            self, side:str, 
            transform:transforms=transforms.ToTensor()
            ):
        return ImageDataset(self.__dict__[side], transform)

    def __str__(self):
        s = ""
        for name in self.images.keys():
            s += f"{name} {len(self.images[name])}\n"
        return s

class DataLoader:
    """
    This class provides functions to load images from datasets. 
    Available datasets are 'CIFAR10' and 'CelebA'.

    Args:
        dataset: name of the dataset
        batch_size: number of images in each batch
        num_blocks: number of blocks of generator and discriminator
        device: device that runs the model
        data_root: path containing the dataset. 
            if dataset is not already downloaded, 
            class downloads them automatically

    NOTE: IF YOU CAN NOT DOWNLOAD/USE CELEBA DATASET, CHECK README FILE. 
    """
    def __init__(
            self, dataset:str, batch_size=32, 
            num_blocks=9, device="cuda", 
            data_root="data/", 
            is_pickle=True):
        available_datasets = ["CIFAR10", "CelebA", "Custom"]

        err = "invalid dataset. available datasets: " + ", ".join(available_datasets)
        assert dataset in available_datasets, err

        image_size = 4 * 2 ** (num_blocks - 1)
        self.num_blocks = num_blocks
        self.device = device
        self.dataset = dataset

        if dataset=="CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

            ds1 = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
            ds2 = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
            ds = torch.utils.data.ConcatDataset([ds1, ds2])
        elif dataset=="CelebA":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

            ds = torchvision.datasets.CelebA(root=data_root, split='all', download=True, transform=transform)
        elif dataset=="Custom":
            # use costom data loader 
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
            imgs = Images([data_root], ["data"], filter, is_pickle)
            ds = imgs.get_dataset("data", transform)
        self.ds_len = len(ds)
        self.dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    def load_images(self):
        """
        This function returns a batch of images. 
        Since the model uses different sized versions of the same image,
        we resize and return a dictionary, whose batch with index i 
        corresponds to  i-th generator/discriminator block.
        """
        x = OrderedDict()
        if self.dataset != "Custom":
            imgs, _ = self.dataloader._get_iterator().__next__()
        else:
            imgs = self.dataloader._get_iterator().__next__()
        imgs = imgs.to(self.device)
        
        for i in range(self.num_blocks-1, -1, -1):
            x[i] = imgs
            imgs = avg_pool2d(imgs, kernel_size=2)
        
        return x

    def get_len(self):
        """
        This function returns the size of the dataset.
        """
        return self.ds_len