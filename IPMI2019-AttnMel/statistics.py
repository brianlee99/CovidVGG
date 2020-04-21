# compute standard deviation and mean for all images
import os
import csv
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.utils as utils
import torchvision.transforms as torch_transforms

from networks import AttnVGG, VGG
from loss import FocalLoss
from data import preprocess_data, ISIC
from utilities import *
from transforms import *


# https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/2

transform_train = torch_transforms.Compose([
     RatioCenterCrop(0.8),
     Resize((256,256)),
     RandomCrop((224,224)),
     RandomRotate(),
     RandomHorizontalFlip(),
     RandomVerticalFlip(),
     ToTensor()
])
transform_val = torch_transforms.Compose([
     RatioCenterCrop(0.8),
     Resize((256,256)),
     CenterCrop((224,224)),
     ToTensor()
])
transform_test = torch_transforms.Compose([
     RatioCenterCrop(0.8),
     Resize((256,256)),
     CenterCrop((224,224)),
     ToTensor()
])

# Set the seed for reproducible results
def _worker_init_fn_():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

def main():
    trainset = ISIC(csv_file='train.csv', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=8, worker_init_fn=_worker_init_fn_())
    valset = ISIC(csv_file='val.csv', transform=transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, num_workers=8)
    testset = ISIC(csv_file='test.csv', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=8)

    mean = 0.0
    std = 0.
    nb_samples = 0.
    for loader in [trainloader, valloader, testloader]:
        for imgs in loader:
            data = imgs['image']
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples


    print("Mean:", mean)
    print("Standard deviation:", std)

    # Mean: tensor([0.5500, 0.5506, 0.5520])
    # Standard deviation: tensor([0.1788, 0.1786, 0.1787])

if __name__ == '__main__':
    main()
