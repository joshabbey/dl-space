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
#import helper


### Get Dataset ###

# Change directory for ship images
os.chdir("Y:\Programmes\SpAIce\Ship Detection\Data\Airbus")

print (os.listdir('.'))

# Assign spreadsheet file name tol 'file'
file = 'train_ship_segmentations_v2.xlsx'

# Load spreadsheet data into class vector
class_data = pd.read_excel(file)

classes = (class_data['Class'])

class_tensor = torch.FloatTensor(classes)

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print ('CUDA is not available. Training on CPU...')
else: 
    print ('CUDA is available! Training on GPU...')
    
# Load and prepare the data

# Number of subprocesses to use for data loading
num_workers = 0
# How many samples per batch to load
batch_size = 20
# Percentage of training set to use as validation
valid_size = 0.2

data_dir_train = 'Train'

data_dir_test = 'Test'

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = datasets.ImageFolder(data_dir_train,transform = transform)

test_data = datasets.ImageFolder(data_dir_test,transform = transform)

# Will also need to access excel file and create a tensor of the labels for loss calculation

# Obtain training indices that will be used for validation

num_train = len(train_data)

indices = list(range(num_train))

# Can't shuffle as we need to track against tensor from excel

split = int(np.floor(valid_size*num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size, sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers)
test_loader =  torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = num_workers)

"""
### Visualise a batch of training data ###

# helper function to (unnormalise and) display an image

def imshow(img):
    #img = img / 2 + 0.5 # unnormalize    
    plt.imshow(np.transpose(img,(1,2,0))) # convert from Tensor Image

# obtain one batch of training images
dataiter = iter(train_loader)
images,labels = dataiter.next()

#print (images.size())

images = images.numpy() #convert images to numpy for display

# Plot the images in the batch, along with corresponding label

fig = plt.figure(figsize =(25,4))

#display images

for idx in np.arange(0):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
  

# Images are 768*768*3 (RGB)
"""

### Define the CNN architecture ###
 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        #convolutional layer 1 (sees 768*768*3)
        self.conv1 = nn.Conv2d(3,8,3,padding=1)
        
        #convolutional layer 2 (sees 192*192*8)
        self.conv2 = nn.Conv2d(8,8,3,padding=1)
        
        # convolutional layer 3 (sees 48*48*8)
        self.conv3 = nn.Conv2d(8,16,3,padding=1)    
                        
        # max pooling layer
        self.pool = nn.MaxPool2d(4,4)

        # linear layer 1 (8*12*12 -> 250)        
        self.fc1 = nn.Linear(16*12*12,250)
        
        #linear layer 2 (250 -> 1)
        self.fc2 = nn.Linear(250,1)
        
        # Dropout layer (p =0.25)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self,x):
        # Add a sequence of convolutional and max pooling layers        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten image output
        x = x.view(-1,16*12*12)
        
        # Add a dropout layer
        #x = self.dropout(x)
        
        # Add first hidden layer, with relu activation
        x = F.relu(self.fc1(x))
        
        # Add a second dropout layer
        #x = self.dropout(x)
        
        # Add second hidden layer, with relu activation
        x = F.relu(self.fc2(x))
        
        return x
    

# Create instantiation of CNN
model = Net()
#print(model)

# Specify loss function (categorical cross-entropy)
criterion = nn.MSELoss()

train_no = 20
valid_no = 10

# Specify Optimiser 
optimiser = optim.SGD(model.parameters(),lr = 0.01)

# Number of epochs to train the model
n_epochs = 10

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
    
    for q in range(train_no):
    #for data, target in train_loader:    
                
        data, labels = dataiter.next()
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, class_tensor = data.cuda(), class_tensor.cuda()
        
        # Clear the gradients of all optimised variables
        optimiser.zero_grad()      
                
        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        # Calculate the batch loss    
                
        loss = criterion(output, class_tensor[i:i+batch_size]) # This may not work       
                
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
        
        data,labels = dataiter.next()
        
        if train_on_gpu:
            data,class_tensor = data.cuda(), class_tensor.cuda()
        
        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        # Calculate the batch loss        
        loss = criterion(output, class_tensor[i:i+batch_size])
        
        # Update average validation loss
        valid_loss += loss.item()*data.size(0)         
    
    
    # calculate average losses
    #train_loss = train_loss/len(train_loader.dataset)
    #valid_loss = valid_loss/len(valid_loader.dataset)

    train_loss = train_loss/(batch_size*train_no)
    valid_loss = valid_loss/(batch_size*valid_no)      

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))  
      
        
        
        
        
        
        
        
        
    




 























