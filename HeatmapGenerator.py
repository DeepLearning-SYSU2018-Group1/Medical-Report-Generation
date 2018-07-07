import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

class HeatmapGenerator():

    def __init__(self, extractor, transform):
        # extractor = self.extractor
        # self.extractor.model caishi model
        self.extractor = extractor
        self.model = self.extractor.model
        self.model.eval()
        #self.features_num = len(list(self.model.parameters())[-1])
        self.transform = transform

        #---- Initialize the batch normalization weights
        self.weights = list(self.model.parameters())[-2]
    
    def generate_heatmap(self, ImagePath, OutputHeatmapPath, transCrop):
        self.get_visual_features(ImagePath, OutputHeatmapPath, transCrop)
        self.get_heatmap(ImagePath, OutputHeatmapPath, transCrop)

    def generate_heatmap_when_training(self, input_visual_features, ImagePath, OutputHeatmapPath, transCrop):
        #self.get_visual_features_when_training(ImagePath, OutputHeatmapPath, transCrop)
        self.visual_features = input_visual_features
        print('self.visual_features.size()',self.visual_features.size())
        self.get_heatmap(ImagePath, OutputHeatmapPath, transCrop)

    def get_visual_features(self, ImagePath, OutputHeatmapPath, transCrop):
        print('get visual features heatmap')

        imageData = Image.open(ImagePath).convert('RGB')
        imageData = self.transform(imageData)
        # 增加batch_size维度
        imageData = imageData.unsqueeze_(0)

        self.model.cuda()

        # with torch.no_grad:
        # self.visual_features, _ = self.extractor.forward(imageData.cuda())
        self.visual_features = self.model(imageData.cuda())
        print('self.visual_features.size()',self.visual_features.size())

    def get_heatmap(self, ImagePath, OutputHeatmapPath, transCrop):
        print('generate heatmap')

        #---- Generate heatmap
        heatmap = None
        for i in range(0, len(self.weights)):
            map = self.visual_features[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map

        #for i in range(0, len(self.weights)):
            #if i == 0: heatmap = visual_features[0,i,:,:]
            #else: heatmap += visual_features[0,i,:,:]

        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(ImagePath, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))

        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

        img = heatmap * 0.5 + imgOriginal

        cv2.imwrite(OutputHeatmapPath, img)

#h = HeatmapGenerator(pathModel, transform)
#h.generate(pathInputImage, pathOutputImage, transCrop)
