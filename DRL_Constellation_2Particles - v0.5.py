# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 07:14:54 2018

@author: z3332903
"""

# begin with two particles that move around in a one-dimensional loop
# the state vector should be a 4*1 initially: p1,v1,p2,v2
# the action vector will be a 2*1 - a1,a2
# reward based on abs(distance-desiredDistance) and perhaps later on, abs(accelerations)
# if abs(distance-desiredDistance) < threshold, reward = 1, else reward = -abs(distance-desiredDistance)
# if reward, then = reward - abs(accelerations)*constant perhaps
# add in some randomness to actuation and velocity eventually
# objects: network, agents, environment
# agents have a state, choose actions, these actions along with the state are then parsed to 
#the environment where the state is updated
# this new state should then be evaluated against the old state within the agent object, this will determine reward/loss
# this loss (along with the action vector?) is sent to the network to back propogate(how?)
# these new weights and biases (decision matrix) are multiplied with the updated state and predict the new actions

# assume time step of 1 sec for all

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy.random
import random
import matplotlib
from matplotlib import pyplot as plt

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print ('CUDA is not available. Training on CPU...')
else: 
    print ('CUDA is available! Training on GPU...')

### the list of all possible actions. accelerations are binary not continous. A continous action space is the realm of an policy network
actions = torch.tensor([[0.1,0.1],[0,0],[-0.1,-0.1],[0.1,0],[0,0.1],[-0.1,0],[0,-0.1],[0.1,-0.1],[-0.1,0.1]])

act = 0 # initialise act - from 0-8

action = 0 # initialise action vector - action = actions[act]

r = 0 # initialise reward

# Two networks will need to be created - one the target, the other the updated
# Initialise the networks to start as the same

maxQ1 = 0 # initiliase maxQ 

targetQ = torch.tensor([0,0,0,0,0,0,0,0,0]) # initialise target q

reward_discount = 0.99

reward_counts = [] # initialise reward vector

delta = 0 # delta between states

target_delta = 0 # target distance

delta_threshold = 1 # threshold distance between agents

lr = 0.001 # learning rate

randomChance = 0.95 # to add some stochastic behaviour

randomSet = torch.tensor([[0.1,0.1],[0,0],[-0.1,-0.1],[0.1,0],[0,0.1],[-0.1,0],[0,-0.1],[0.1,-0.1],[-0.1,0.1]])

memories = [] # all previous memories

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def discount_rewards(rewards):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):              
        running_add = running_add * reward_discount + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

def update_state(state,action):
            
    # Stochastically update velocities
    """    
    if np.random.rand(1) < randomChance: 
        new_state[1] = state[1] + action[0] 
    else:
        new_state[1] = state[1] + action[0] + numpy.random.choice(randomSet)        
    """             
       
    new_state = torch.tensor([0.0,0.0,0.0,0.0])
    delta = 0
    
    #update velocities
    new_state[1] = state[1] + action[0]
    new_state[3] = state[3] + action[1]    
    
    #update positions
    new_state[0] = (state[0] + new_state[1]) %10
    new_state[2] = (state[2] + new_state[3]) %10
    
    state = new_state
    
    delta = min(abs(state[0] - state[2]), abs((state[0]-10) - (state[2])),abs((state[0]) - (state[2]-10)))
    
    #delta = abs(state[0] - state[2])  
                      
    return state, delta
    

def get_reward(delta,state):
    
    r = 0        
    
    if delta <= 1.2 and delta >= 0.8:
        r = 0.1
                      
    return r


### This is where the nuts and bolts of DQL happens - need to check this against other code
def experience_replay(q,q1,r,act):               
           
    targetQ = torch.tensor([0,0,0,0,0,0,0,0,0])
    
    targetQ = targetQ.float()

    for i in range(len(targetQ)):
        targetQ[i] = q1[i]*reward_discount + r            
                        
    return targetQ

### Define the NN architecture ###
 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        # state input of size 2               
        # linear layer 1 (2 -> 30)        
        self.fc1 = nn.Linear(4,30)       
                
        #linear layer 2 (30 -> 20)
        self.fc2 = nn.Linear(30,20) 
                
        #linear layer 3 (20 -> 9)
        self.fc3 = nn.Linear(20,9)        
        
        # output of size 3 - the quality of each move
        
        # Dropout layer (p =0.25)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):              
                
        # Add input layer to first hidden layer, with relu activation
        x = F.relu(self.fc1(x))
        
        # Add a first dropout layer
        x = self.dropout(x)
        
        # Add first hidden layer to second hidden layer, with relu activation
        x = F.relu(self.fc2(x))
        
        # Add a second dropout layer
        x = self.dropout(x)
        
        # Add second hidden layer to output layer
        x = self.fc3(x)               
               
        return x
    

# Create instantiation of NN - one for updating the other as a target
model_update = Net()
model_target = model_update

# Specify loss function (mean squared error loss)
criterion = nn.MSELoss()

# Specify Optimiser 
optimiser = optim.SGD(model_update.parameters(),lr = 0.001,momentum=0.1) # can add momentum if needed

# Number of epochs to train the model
n_epochs = 10

#########################
### Train the network ###
######################### 


for epoch in range(n_epochs): 
    
    model_target = model_update

    t = 0
    
    losses = []

    averages = []

    e = 0.75 # e-greedy exploration factor - this starts low and gets higher the longer the algorithm runs
        # more exploration early, exploitation later        
    
    for games in range(100):
    
        j = 0 # Moves counter
        
        totalReward = 0 # Reward counter
        
        state = torch.tensor([0.0,0.0,0.0,0.0]) # [current position, current velocity]

        state = state.float()          
        
        memories_recent = [] # reset memories       
                
        rewards = [] # for discounting
        
        discounted_rewards = [] # discounted rewards
                
        delta = 0 # reset         
        
        while j <50 and delta <1.5:        
               
            r = 0 # reset reward
            
            # feed forward         
                                
            q = model_update(state)                                            
           
            #print()
            if np.random.rand(1) < e: # e-greedy action selection
                act = torch.argmax(q) # need to get this to work with tensors
            else:
                act = np.random.randint(0,2)                         
            
            action = actions[act] # action to take                                   
            
            stateOld = state # keep old state for memory                  
                    
            state, delta =  update_state(state,action) # update state and calculate delta

            #print(state)            
                                
            r = get_reward(delta,state) # get reward                             
            
            q1 = model_target(state) #re-calc new q values based on target network          
                                                   
            memories_recent.append([stateOld,q1,r,state,q,act]) # record memories for experience replay               
    
            rewards.append(r)
            
            totalReward += r
                
            j+= 1
            
            if e < 0.99:
            
                e += 0.001                            
           
        discounted_rewards = discount_rewards(rewards) # apply discount to the rewards vector
        
        for i in range(len(memories_recent)):
            memories_recent[i][2] = discounted_rewards[i]
        
        ### check memory importance
        
        important = 0
        
        if memories_recent[0][2] > 0:
            important = 1                    
                           
        random.shuffle(memories_recent) # get rid of any correlation between memories
            
        # Store memories from most recent game              
                               
        for b in range(len(memories_recent)): # all recent memories used
                    
            batch = memories_recent[b] # retrieve memory
    
            memories.append(batch) # add to whole selection of memories
            
        ###Update based on the whole of the most recent game if it was a good game###
            
        if important ==1:
            for b in range(len(memories_recent)): # all recent memories
                batch = memories_recent[b] # retrieve memory
                
                target = experience_replay(batch[4],batch[1],batch[2],batch[5]) # calculate target for memory
            
                # Prediction from memory
                prediction = batch[4]          
                            
                # move tensors to GPU if CUDA is available
                #if train_on_gpu:
                #   prediction, target, model_update = prediction.cuda(), target.cuda(), model_update.cuda()
        
                # Clear the gradients of all optimised variables
                optimiser.zero_grad()            
            
                # Calculate the batch loss                        
                loss = criterion(prediction,target)                                             
                                  
                # Backward pass: compute the gradient of the loss with respect to model parameters
                loss.backward(retain_graph=True)
        
                # Perform a single optimisation step
                optimiser.step()

                # Track Losses
                losses.append(loss)               
                        
                t += 1
        
        ### Update based on whole of game experiences ###
                
        else:          
         
            for i in range(len(memories_recent)): # all recent memories used   

                model_update.train()                                             
                    
                b = random.randint(0,len(memories_recent)-1) # select a random memory
            
                batch = memories[b] # retrieve memory        
                     
                target = experience_replay(batch[4],batch[1],batch[2],batch[5]) # calculate target for memory
            
                # Prediction from memory
                prediction = batch[4]            
                            
                # move tensors to GPU if CUDA is available
                #if train_on_gpu:
                #   prediction, target, model_update = prediction.cuda(), target.cuda(), model_update.cuda()
        
                # Clear the gradients of all optimised variables
                optimiser.zero_grad()            
            
                # Calculate the batch loss                        
                loss = criterion(prediction,target)                                                            
                                  
                # Backward pass: compute the gradient of the loss with respect to model parameters
                loss.backward(retain_graph=True)
        
                # Perform a single optimisation step
                optimiser.step()

                # Track Losses
                losses.append(loss)               
                        
                t += 1        
            
        # update target network
        if t > 1000:
            model_target = model_update
            
            t =0      
        
        reward_counts.append((totalReward))        
             
            
    averages.append(numpy.average(reward_counts))    
   
    print(numpy.average(averages))
    
    #plt.plot(losses)
    #plt.show()
    
    
    

    
    
    
    