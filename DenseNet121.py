import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet121, self).__init__()

        # CheXNet is a 121-layer Dense Convolutional Net- work (DenseNet) (Huang et al., 2016)
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        # We replace the final fully connected layer with one that has a single output
        # after which we apply a sigmoid nonlinearity.
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward(self, x):
        x = self.densenet121(x)
        return x
