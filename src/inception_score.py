import torch
from torch.nn import Module
from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from math import ceil

class Inception_Score(Module):
    """Class to compute Inception Score"""
    def __init__(self, device):
        super(Inception_Score, self).__init__()
        self.model = inception_v3(pretrained=True, transform_input=True).to(device)
        self.model.eval()
        #self.preprocess =  # we need to preprocess images before we feed them to the inception_v3 network 
        self.device = device
    
    def forward(self, images_set, batch_size=32, num_splits=10):
        loader = DataLoader(images_set, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            all_preds = torch.zeros((len(images_set), 1000))
            ind = 0
            for images_batch in loader:
                images_batch = images_batch.to(self.device)

                #images_batch = self.preprocess(images_batch)
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