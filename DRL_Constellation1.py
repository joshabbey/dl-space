# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 07:14:54 2018

@author: z3332903
"""

# start wtih two particles that move around in a circle
# the state vector should be a 4*1 intially, p1,v1,p2,v2
# the action vector will be a 2*1 - a1,a2
# reward based on abs(distance-desiredDistance) and abs(accelerations)
# if abs(distance-desiredDistance) < threshold, reward = 1, else reward = -abs(distance-desiredDistance)
# reward then = reward - abs(accelerations)*constant perhaps
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

actions = [[0.1,0.1],[0,0],[-0.1,-0.1],[0.1,0],[0,0.1],[-0.1,0],[0,-0.1],[0.1,-0.1],[-0.1,0.1]]

act = 0

action = []

r = 0 

state = [5,0,0,0,0,0] # p1,v1,a1,p2,v2,a2

#w = numpy.ones((len(actions),len(state)))

#w = np.array([[0.2,0.1,0,-1,-0.5,0.04],[0.3,0.2,-1.007,0.06,-0.61,0.6],[-0.3,0.1,-1.2,-0.87,-0.1,-0.4],[0.4,-0.8,0.07,1.4,-1.2,-0.45],[0.6,-0.32,0.11,-0.26,0.1,0.01],[0.08,-0.4,-0.1,-0.8,0.3,0.1],[0.01,-0.023,0.4,0.2,-0.17,-0.1],[-1,1,-0.19,0.67,-0.23,-0.05],[0.11,-0.21,0.14,-0.08,0,0.05]])

w = np.random.normal(0,1,(len(actions),len(state)))

q1 = [0,0,0,0,0,0,0,0,0]

maxQ1 = 0

targetQ = [0,0,0,0,0,0,0,0,0]

loss_delta = [] # loss delta - Qt - Q1

loss = [] # 1/2(loss_delta)**2

b = numpy.zeros((len(state),len(actions)),) # back matrix

gamma = 0.95 # discount

reward_counts = []

delta = 0 # delta between states

previous_delta = abs(state[0] - state[3]) # previous delta between states. to compare how the agents are going

target_delta = 0 # target distancebetween agents

delta_threshold = 1 # threshold distance between agents

e = 0.4 # e-greedy exploration factor - this starts low and gets higher the longer the algorithm runs
# more exploration early, exploitation later

lr = 0.003 # learning rate

def discount_rewards(rewards):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):              
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

def update_state(state,action):
    
    new_state = [0,0,0,0,0,0]
    delta = 0
    
    #update accelerations
    new_state[2] = state[2] + action[0]
    new_state[5] = state[5] + action[1]
    
    #update velocities
    new_state[1] = state[1] + new_state[2] 
    new_state[4] = state[4] + new_state[5]
    
    #update positions
    new_state[0] = (state[0] + new_state[1]) #%100
    new_state[3] = (state[3] + new_state[4]) #%100
    
    state = new_state
    
    #delta = min(abs(state[0] - state[3]), abs((state[0]-100) - (state[3])),abs((state[0]) - (state[3]-100)))
    
    delta = abs(state[0] - state[3])    
            
    # need to update this with a new delta check for 99 to 2 style
    
    return state, delta
    

def get_reward(delta,previous_delta):
    
    r = 0
    
    if delta <= (target_delta + delta_threshold) and delta >= (target_delta - delta_threshold):
        r = 1      
    elif delta < previous_delta: 
        r = 0.1
    else:
        r = -0.1
        
        #this is only if time matters, otherwise should it be punished?    
           
    return r

def back_propogate(state,loss_delta,w):
    
    b = numpy.zeros((len(state),len(actions)),) # backprop matrix

    #print (loss_delta)         
                        
    b = np.outer(loss_delta,state) # dl/dw = dl/dq * dq/dw - you should get this to work as the other method is computational inefficient!
    
    #print(w)
        
    b = lr*b
    
    print (b)
    print('next')
            
    w = np.add(w,b) # add new weights times learning rate

    #print(np.subtract(w,lr*b))        
        
    return w

def experience_replay(q,q1,r):
    
    maxQ1 = numpy.max(q1) # set max q1 - target value
                
    targetQ = q1 # ready the target vector
        
    targetQ[0:9] = (r + gamma*maxQ1) # set the target vector
    
    #print (targetQ)         
            
    loss_delta = np.subtract(targetQ,q) # delta Q
    
    #print (loss_delta)
                       
    #loss = (1/2)*(loss_delta)**2 # calculate loss vector   
       
    return loss_delta

#### Run training #####

for c in range(1):

    j = 0 # counter
    
    totalReward = 0 # reward counter
    
    state = np.array([5,0,0,2,0,0])
    
    memories = [] # reset memories
    
    rewards = [] # for discounting
    
    discounted_rewards = [] # discounted rewards
    
    previous_delta = 3
    
    #while delta <= 10:
    for t in range(10):
           
        r = 0 # reset reward
        
        q = w.dot(state) #re-calc q values
        
        print (state)
        print()
        print (w)
        print()
        print(q)
        print()
        if np.random.rand(1) < e: # e-greedy action selection
        
            act = np.argmax(q)
        else:
            act = np.random.randint(0,9)            
        
        action = actions[act] # action to take
        
        print(act)
        print()        
        print (action)
        print ()
        
        stateOld = state # keep old state for memory        
                
        state, delta =  update_state(state,action) # update state and calcualte delta       
                
        r = get_reward(delta,previous_delta) # get reward
        
        print (r)
        print()
        
        previous_delta = delta # update previous delta for next iteration
                
        q1 = w.dot(state) # get new predicted q values from new state                  
                        
        memories.append([stateOld,q1,r,state,q]) # record memories for experience replay               

        rewards.append(r)       
                
        """
        ##### the following should probably be in the experience replay back prop function##########
        
        maxQ1 = numpy.max(q1) # set max q1 - target value
        
        targetQ = q # ready the target vector
        
        targetQ[0:8] = r + y*maxQ1 # set the target vector
        
        loss_delta = targetQ-q1 # delta Q
        
        loss = (1/2)*(loss_delta)**2 # calculate loss vector
        
        # now back propogate the loss to the weights      
                
        w = back_propogate(state,loss_delta,w)
        
        #############################################################################################
        """                
                
        if j % 20 == 0 and j != 0: # experience replay every 50 steps
            for b in range(10):
                i = random.randint(0,len(memories) -1)                              
                
                batch = memories[i] # retrieve memory               
                                                
                discounted_rewards = discount_rewards(rewards) # apply discount to the rewards vector                                                        
                
                loss_delta = experience_replay(batch[4],batch[1],discounted_rewards[i]) # calculate loss vector for memory              
                                                   
                w = back_propogate(batch[0],loss_delta,w) # back rpop using memory (s0,loss,weights)
        elif j< 30: # for some immediate changes
            
            loss_delta = experience_replay(q,q1,r)
            
            w = back_propogate(stateOld,loss_delta,w)
            
        
                
        totalReward += r
            
        j+= 1
        
        if e < 0.99:
        
            e += 0.01
        
        #print (state)
        #print(r)
      
    #print(rewards)
    #print (discounted_rewards)
    print (totalReward)
    print()
    
    reward_counts.append((totalReward))
          
       
        
    
#plt.plot(reward_counts)
#plt.show()

    
    
    
    