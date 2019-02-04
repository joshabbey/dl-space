# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 09:56:03 2019

@author: z3332903
"""
import os
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFile
import json

# to help with truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

#import helper

### Get Dataset ###

# Change directory for json
os.chdir("Y:\Programmes\SpAIce\Ship Detection\Data\Planet")

# open json file
with open('shipsnet.json') as json_file:
    class_data = json.load(json_file) 

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print ('CUDA is not available. Training on CPU...')
else: 
    print ('CUDA is available! Training on GPU...')
    
# Load and prepare the data
    
# Change directory for ship images
os.chdir("Y:\Programmes\SpAIce\Ship Detection\Data\Planet")

# Number of subprocesses to use for data loading
num_workers = 0
# How many samples per batch to load
batch_size = 1
# Percentage of training set to use as validation and testing
valid_size = 0.1
# percentage of training set to use for testing - as the test data isn't labelled!
test_size = 0.1

data_dir_train = 'shipsnet'

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = datasets.ImageFolder(data_dir_train,transform = transform)

print (train_data.__len__())

# Obtain training indices that will be used for validation

num_train = len(train_data)

indices = list(range(num_train))

# Can't shuffle as we need to track against tensor from excel

split_valid = int(np.floor((1-valid_size - test_size)*num_train))
split_test =  int(np.floor((1 - test_size)*num_train))
train_idx, valid_idx, test_idx = indices[:split_valid], indices[split_valid:split_test], indices[split_test:]

# Define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(valid_idx)

print(len(train_idx))
print(len(valid_idx))
print(len(test_idx))

# Prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size, sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers)
test_loader =  torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler = test_sampler, num_workers = num_workers)

### Visualise a batch of training data ###

# helper function to (unnormalise and) display an image

def imshow(img):
    plt.imshow(img)

# obtain one batch of training images
dataiter = iter(train_loader)
image,label = dataiter.next()

print (image[0].size())

image = image[0][1].numpy() #convert images to numpy for display

# Plot the images in the batch, along with corresponding label
fig = plt.figure()

#display image
imshow(image)

# Images are 80*80*3 (RGB)

"""

### Define the CNN architecture ###
 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        #convolutional layer 1 (sees 80*80*3)
        self.conv1 = nn.Conv2d(3,64,5,padding=2)
        
        #convolutional layer 2 (sees 768*768*64)
        self.conv2 = nn.Conv2d(64,32,3,padding=1)
        
        # convolutional layer 3 (sees 128*128*32)
        self.conv3 = nn.Conv2d(32,32,5,padding=2)
        
        # convolutional layer 4 (sees 128*128*32)
        self.conv4 = nn.Conv2d(32,16,3,padding=1) 
                        
        # max pooling layer 1
        self.pool1 = nn.MaxPool2d(3,3)
        
        # max pooling layer 2
        self.pool2 = nn.MaxPool2d(4,4)

        # linear layer 1 (16*12*12 -> 750)        
        self.fc1 = nn.Linear(16*16*16,750)
        
        #linear layer 2 (250 -> 1)
        self.fc2 = nn.Linear(750,100)
        
        #linear layer 3 (250 -> 1)
        self.fc3 = nn.Linear(100,1)
        
        # Dropout layer (p =0.25)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self,x):
        # Add a sequence of convolutional and max pooling layers        
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
                
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        
        # Flatten image output
        x = x.view(-1,16*16*16)
        
        # Add a dropout layer
        x = self.dropout(x)
        
        # Add first hidden layer, with relu activation
        x = F.relu(self.fc1(x))
        
        # Add a second dropout layer
        x = self.dropout(x)
        
        # Add second hidden layer, with relu activation
        x = F.relu(self.fc2(x))
        
        # Add a third dropout layer
        x = self.dropout(x)
        
        # Add third hidden layer, with relu activation
        x = F.relu(self.fc3(x))
        
        return x
    

# Create instantiation of CNN
model = Net()
#print(model)

# Specify loss function (categorical cross-entropy)
criterion = nn.MSELoss()

train_no = 25000 #7700
valid_no = 1000 #960

# Specify Optimiser 
optimiser = optim.SGD(model.parameters(),lr = 0.01)

# Number of epochs to train the model
n_epochs = 1

valid_loss_min = np.Inf # Track change in validation loss

dataiter = iter(train_loader)

### Train the network ###

for epoch in range(1, n_epochs+1):
    
    # Keep track of training and validation loss
    
    train_loss = 0
    valid_loss = 0
    
    ### Train the model ###
    
    model.train()
    i = 0
    j = split_valid
    
    for q in range(train_no):
    #for data, target in train_loader:

        print(q)
        
        dataiter = iter(train_loader)        
                
        data, labels = dataiter.next()
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, class_tensor, model = data.cuda(), class_tensor.cuda(), model.cuda()
        
        # Clear the gradients of all optimised variables
        optimiser.zero_grad()             
                        
        # Forward pass: compute predicted outputs by passing inputs to the model
        
        output = model(data)

        print(output)
        print(class_tensor[i:i+batch_size])        
                        
        # Calculate the batch loss    
                
        loss = criterion(output, class_tensor[i:i+batch_size])     
                        
        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimisation step
        optimiser.step()
        
        # Update training loss
        
        train_loss += loss.item()*data.size(0)
        
        i += batch_size        
      
    
    ### Validate the model ###
    model.eval()    
    for q in range(valid_no):      
                
        print(q)
        
        dataiter = iter(valid_loader)
        
        data,labels = dataiter.next()
        
        if train_on_gpu:
            data,class_tensor, model = data.cuda(), class_tensor.cuda(), model.cuda()          
             
        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        print(output)
        print(class_tensor[j:j+batch_size])         
        
        # Calculate the batch loss        
        loss = criterion(output, class_tensor[j:j+batch_size])
        
        # Update average validation loss
        valid_loss += loss.item()*data.size(0)

        j += batch_size           
    
    
    # calculate average losses
    #train_loss = train_loss/len(train_loader.dataset)
    #valid_loss = valid_loss/len(valid_loader.dataset)

    train_loss = train_loss/(batch_size*train_no)
    valid_loss = valid_loss/(batch_size*valid_no) 
       
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))  
      
        
        
"""        
        
        
        
        
        
    




 























