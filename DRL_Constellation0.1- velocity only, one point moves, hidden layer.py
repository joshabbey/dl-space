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

import numpy as np
import numpy.random
import random
import matplotlib
from matplotlib import pyplot as plt

### the list of all possible actions. accelerations are binary not continous. A continous action space is the realm of an actor critic network
actions = [[0.1],[0],[-0.1]]

act = 0 # initialise act - from 0-2

action = [] # initialise action vector - action = actions[act]

r = 0 # initialise reward

state = np.array([0,0]) #declaring state of the two particles vector. p1,v1,p2,v2

hidden = np.array([0,0,0,0,0,0,0,0,0,0]) #declaring hidden layer

w1 = np.random.normal(0,0.1,(len(hidden),len(state))) # initialise weights in the network
#w1 = numpy.ones((len(hidden),len(state)))

w2 = np.random.normal(0,0.1,(len(actions),len(hidden))) # initialise weights in the network
#w2 = numpy.ones((len(actions),len(hidden))) # initialise weights in the network

w1Target = w1

w2Target = w2

q1 = [0,0,0] # initialise q1 vector - predictor of the quality of a move

maxQ1 = 0 # initiliase maxQ 

targetQ = [0,0,0] # initialise target q

loss_delta = [] # loss delta - targetQ - q1

loss = [] # 1/2(loss_delta)**2

b = numpy.zeros((len(state),len(actions)),) # initialise back propogration matrix

reward_discount = 0.99

reward_counts = [] # initialise reward vector

j_counts = [] # how long did it last?

delta = 0 # delta between states

#previous_delta = abs(state[0] - state[3]) # previous delta between states. to compare how the agents are going

target_delta = 0 # target distance between agents

delta_threshold = 1 # threshold distance between agents

e = 0.5 #0.94 # e-greedy exploration factor - this starts low and gets higher the longer the algorithm runs
# more exploration early, exploitation later

lr = 0.000001 # learning rate

randomChance = 0.9 # to add some stochastic behaviour

randomSet = [0.1,0,-0.1]

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
    
    new_state = [0,0]  #[0,0,state[0],state[1]]
    delta = 0
    
    #update velocities - with some randomness
    
    """
    if np.random.rand(1) < randomChance: 
        new_state[1] = state[1] + action[0] 
    else:
        new_state[1] = state[1] + action[0] + numpy.random.choice(randomSet)
    """              
             
    new_state[1] = state[1] + action[0]       
        
    #update positions
    #new_state[0] = (state[0] + new_state[1]) %10
    new_state[0] = (state[0] + new_state[1])    
    
    state = new_state
    
    #delta = min(abs(state[0] - 0), abs((state[0]-10) - 0),abs((state[0]) - (-10)))
    
    delta = abs(state[0])
    
    #delta = abs(state[0] - state[2]) 
              
    return state, delta
    

def get_reward(delta,state):
    
    r = 0
          
    # short term approach to test if it is starting to learn
    
    """
    if delta <= (target_delta + delta_threshold) and delta >= (target_delta - delta_threshold):
        r = 1      
    elif delta < previous_delta: 
        r = 0.1
    else:
        r = -0.1
    """
    
    # long term approach   
    """"    
    if  abs(delta) < 1 and abs(state[1]) < 1:
        r = 1
        if abs(delta) < 0.5 and abs(state[1]) < 0.5:
            r = 2
            if abs(delta) <= 0.1 and abs(state[1]) <= 0.1:
                r = 10
        #if abs(state[1]) < 0.2: 
         #   r = 10
    #elif delta < previous_delta:
     #   r = 0.1
    else:
        r = -0.05 #this is only if time matters, otherwise should it be punished?
    """

    if delta <= 1:
        r = 0.05
    else:
        r = -1
                       
    return r

def back_propogate(state,hidden,loss_delta,w1,w2):
    
    b1 = numpy.zeros((len(state),len(hidden)),) # backprop matrix 1
    b2 = numpy.zeros((len(actions),len(hidden)),) # backprop matrix 2             
                        
    b1 = np.outer(loss_delta,np.transpose(hidden)) # 6*9 matrix dl/dw = dl/dq * dq/dw - you should get this to work as the other method is computational inefficient!
    b2 = np.dot(np.transpose(w2),loss_delta)    
        
    b2 = np.outer(b2,np.transpose(state))    
            
    b1 = lr*b1
    b2 = lr*b2    
    
    w2 = np.subtract(w2,b1)
    w1 = np.subtract(w1,b2) # add new weights times learning rate !    
        
    return w1,w2

def experience_replay(q,q1,r,act):
    
    maxQ1 = numpy.max(q1) # set max q1 - target value
                 
    targetQ = 0 # ready the target vector   
            
    #targetQ[0:3] = (r + gamma*maxQ1) # set the target vector - all actions are updated
    
    targetQ = (r + reward_discount*maxQ1) # set the target vector - only the action chosen is updated
            
    loss = np.subtract(targetQ,q[act]) # delta Q
    
    # this only updates using the one 'loss'
    
    loss_delta = [0,0,0]
    
    loss_delta[act] = loss
        
    #print (loss_delta)    
                       
    #loss = (1/2)*(loss_delta)**2 # calculate loss vector   
       
    return loss_delta

"""Run training"""
averages = []

for epoch in range(30): 
    
    w1Target = w1

    w2Target = w2

    t = 0
    
    losses = []
    
    for games in range(1000):
    
        j = 1 # counter
        
        totalReward = 0 # reward counter
        
        state = np.array([0,0]) # [current position, current velocity, previous position, previous velocity]          
        
        memories_recent = [] # reset memories
        
        rewards = [] # for discounting
        
        discounted_rewards = [] # discounted rewards
                
        delta = 0 # reset
        
        #print(state)
        #print(w1)               
        
        while delta <= 1 and j <100:        
               
            r = 0 # reset reward
            
            # feed forward
            
            hidden = w1.dot(state)                   
                    
            q = w2.dot(hidden) #re-calc q values            
           
            #print()
            if np.random.rand(1) < e: # e-greedy action selection
                act = np.argmax(q)
            else:
                act = np.random.randint(0,2)                         
            
            action = actions[act] # action to take                        
            
            stateOld = state # keep old state for memory
    
            hiddenOld = hidden        
                    
            state, delta =  update_state(state,action) # update state and calculate delta       
                    
            r = get_reward(delta,state) # get reward          
                        
            hidden = w1Target.dot(state)  
            
            q1 = w2Target.dot(hidden) #re-calc q values   
                                        
            memories_recent.append([stateOld,q1,r,state,q,hiddenOld,act]) # record memories for experience replay               
    
            rewards.append(r)
            
            totalReward += r
                
            j+= 1
            
            if e < 0.99:
            
                e += 0.00001                            
           
        discounted_rewards = discount_rewards(rewards) # apply discount to the rewards vector
        
        for i in range(len(memories_recent)):
            memories_recent[i][2] = discounted_rewards[i]       
                     
        random.shuffle(memories_recent) # get rid of any correlation between memories
    
        # update based on most recent game              
                               
        for b in range(len(memories_recent)): # all recent memories used
                    
            batch = memories_recent[b] # retrieve memory
    
            memories.append(batch) # add to whole selection of memories
            
        # update based on whole of game experiences
            
        for i in range(len(memories_recent)): # all recent memories used                                                
                    
            b = random.randint(0,len(memories_recent)-1)
            
            batch = memories[b] # retrieve memory        
                     
            loss_delta = experience_replay(batch[4],batch[1],batch[2],batch[6]) # calculate loss vector for memory

            losses.append(loss_delta)                      
                                                       
            w1,w2 = back_propogate(batch[0],batch[5],loss_delta,w1,w2) # back prop using memory (s0,loss,weights)
            
            t += 1        
            
        # update target network
        if t > 500:
            w1Target = w1
            w2Target = w2
            
            t =0      
        
        reward_counts.append((totalReward))
        #j_counts.append(j)
        
    #plt.plot(losses)
    #plt.show()         
            
    averages.append(numpy.average(reward_counts))        
        
    #plt.plot(j_counts)
    #plt.show()

    print(numpy.average(averages))
    #print(numpy.average(losses))

    
    
    
    