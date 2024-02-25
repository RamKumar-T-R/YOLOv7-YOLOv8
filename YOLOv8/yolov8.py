# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:45:47 2023

@author: Ramkumar
"""

# Importing the required packages
from ultralytics import YOLO

# Creating the model
model = YOLO('yolov8n.pt')

# Predicting using the model
model.predict(source=0, save=True, save_txt=True, conf=0.6, save_crop=True)