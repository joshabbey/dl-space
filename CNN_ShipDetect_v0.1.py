# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 09:56:03 2019

@author: z3332903
"""

import torch
import numpy as np
#import torchvision
#from torchvision import datasets
#import torchvision.tranforms as transforms
#import helper
import matplotlib.pyplot as plt
#import cv2

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print ('CUDA is not available. Training on CPU...')
else: 
    print ('CUDA is available! Training on GPU...')
    
# Load and prepare the data

#data_dir = 'folder/subfolder'

#transform = transforms.Compose() 


