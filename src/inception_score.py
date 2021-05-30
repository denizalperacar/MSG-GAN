import torch
from torch.nn import Module
from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torchvision.transforms.functional import resize
from math import ceil

class Inception_Score(Module):
    """Class to compute Inception Score"""
    def __init__(self, device):
        super(Inception_Score, self).__init__()
        self.model = inception_v3(pretrained=True, transform_input=True).to(device)
        self.model.eval()
        self.device = device
    
    def forward(self, images, batch_size=32, num_splits=10):
        with torch.no_grad():
            all_preds = torch.zeros((len(images), 1000))
            ind = 0
            for idx in range(0, len(images), batch_size):
                images_batch = resize(images[idx:idx+batch_size].to(self.device), [299, 299])

                preds = self.model(images_batch)
                preds = softmax(preds, dim=1)
                all_preds[ind:ind+preds.shape[0]] = preds
                ind = ind + preds.shape[0]
                
            split_results = torch.zeros(num_splits)
            ind = 0
            split_size = ceil(all_preds.shape[0] / num_splits)
            for i in range(num_splits):
                preds = all_preds[ind:ind+split_size]
                ind = ind + split_size
                p_y = torch.mean(preds, dim=0, keepdims=True)
                #split_results[i] = torch.exp(kl_div(preds, p_y, reduction='batchmean'))
                split_results[i] = torch.exp(torch.sum(preds * (torch.log(preds) - torch.log(p_y))) / preds.shape[0])

            return torch.std_mean(split_results)