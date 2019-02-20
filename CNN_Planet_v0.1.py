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
    data = json.load(json_file)  

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print ('CUDA is not available. Training on CPU...')
else: 
    print ('CUDA is available! Training on GPU...')
    
# Load and prepare the data
    
# Change directory for ship images
os.chdir("Y:\Programmes\SpAIce\Ship Detection\Data")

# How many samples per batch to load
batch_size = 1

# Training data set

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_set = datasets.ImageFolder('Planet\shipsnet\Train',transform=transform)

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,shuffle=True)

print (train_set.__len__())

# Validation data set
valid_set = datasets.ImageFolder('Planet\shipsnet\Valid',transform=transform)

valid_loader = torch.utils.data.DataLoader(valid_set,batch_size = batch_size,shuffle=True)

print (valid_set.__len__())

# Test data set
test_set = datasets.ImageFolder('Planet\shipsnet\Test',transform=transform)

test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size,shuffle=True)

print (test_set.__len__())

"""
### Visualise a batch of training data ###

def imshow(img):
    plt.imshow(img)

# obtain one batch of training images
dataiter = iter(train_loader)
images,hey = dataiter.next()

labels = hey.float()

print(labels)
print (images[0].size())
image = (images[0][0]).numpy() #convert images to numpy for display
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
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        
        #convolutional layer 2 (sees 40*40*64)
        self.conv2 = nn.Conv2d(64,32,3,padding=1)
        
        # convolutional layer 3 (sees 20*20*32)
        self.conv3 = nn.Conv2d(32,32,3,padding=1)      
                               
        # max pooling layer 1
        self.pool1 = nn.MaxPool2d(2,2)       
        
        # linear layer 1 (32*10*10 -> 250)        
        self.fc1 = nn.Linear(16*10*10,250)       
                
        #linear layer 3 (250 -> 1)
        self.fc2 = nn.Linear(250,1)
        
        # Dropout layer (p =0.25)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self,x):
        # Add a sequence of convolutional and max pooling layers        
        x = self.pool1(F.relu(self.conv1(x))) 
        x = self.pool1(F.relu(self.conv2(x)))            
        x = self.pool1(F.relu(self.conv3(x)))        
        
        # Flatten image output
        x = x.view(-1,16*10*10)
        
        # Add a dropout layer
        x = self.dropout(x)
        
        # Add first hidden layer, with relu activation
        x = F.relu(self.fc1(x))
        
        # Add a second dropout layer
        x = self.dropout(x)
        
        # Add second hidden layer, with relu activation
        x = F.relu(self.fc2(x))       
               
        return x
    

# Create instantiation of CNN
model = Net()
#print(model)

# Specify loss function (categorical cross-entropy)
criterion = nn.MSELoss()

train_no = int(3200/batch_size) 
valid_no = int(400/batch_size)

# Specify Optimiser 
optimiser = optim.SGD(model.parameters(),lr = 0.01)

# Number of epochs to train the model
n_epochs = 1

valid_loss_min = np.Inf # Track change in validation loss

dataiter = iter(train_loader)

"""

### Train the network ###

for epoch in range(1, n_epochs+1):
    
    # Keep track of training and validation loss    
    train_loss = 0
    valid_loss = 0
    
    ### Train the model ###
    
    model.train()    
    
    for q in range(train_no):
    #for data, target in train_loader:        
        
        dataiter = iter(train_loader)        
                
        data,hey = dataiter.next()
        labels = hey.float()
        
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, labels, model = data.cuda(), labels.cuda(), model.cuda()
        
        # Clear the gradients of all optimised variables
        optimiser.zero_grad()             
                        
        # Forward pass: compute predicted outputs by passing inputs to the model
        
        output = model(data)                                                
                        
        # Calculate the batch loss  
        loss = criterion(output[0],labels) 
                      
        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimisation step
        optimiser.step()
        
        # Update training loss
        
        train_loss += loss.item()*data.size(0)        
               
          
    ### Validate the model ###
    model.eval() # no trainiung
    dataiter = iter(valid_loader)   
    for q in range(valid_no): 
              
                            
        data,hey = dataiter.next()
        labels = hey.float()
        
        if train_on_gpu:
            data,labels, model = data.cuda(), labels.cuda(), model.cuda()          
             
        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)                        
        
        # Calculate the batch loss        
        loss = criterion(output[0], labels)       
                
        # Update average validation loss
        valid_loss += loss.item()*data.size(0)                
        
    

    train_loss = train_loss/(batch_size*train_no)
    valid_loss = valid_loss/(batch_size*valid_no) 
       
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_planet_ship_detection.pt')
        valid_loss_min = valid_loss
    
"""
 ### Test CNN ###

model.load_state_dict(torch.load('model_planet_ship_detection.pt'))

# track test loss
test_loss = 0.0
correct_predictions = 0
total_predictions = 0
      
model.eval()
# iterate over test data

for data, hey in test_loader:
    # move tensors to GPU if CUDA is available    
    label = hey.float()
    if train_on_gpu:
        data, label, model = data.cuda(), label.cuda(), model.cuda() 
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output[0], label)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    
    if output[0] >= 0.5:
        prediction = 1
    else:
        prediction = 0
        
    print(prediction)    
       
    if prediction == label:
        correct_predictions += 1
    total_predictions += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

print('Correct predictions: {:.6f}\n' .format(correct_predictions))
print('Percentage Correct: {:.6f}\n %' .format((correct_predictions/total_predictions)*100))
    
        
        


           
   
       
        
        
        
        
        
    




 























