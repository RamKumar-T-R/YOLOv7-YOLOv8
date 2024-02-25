# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:55:45 2023

@author: Ramkumar
"""

# Importing the required libraries
import cv2
import torch
from super_gradients.training import models
from super_gradients.common.names import Models

# Note: As of now, only YOLOX and PPYOLOE are available

model = models.get(Models.YOLOX_N, pretrained_weights = "coco")

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

