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
actions = [[0,0],[0.1,0],[-0.1,0]]

act = 0 # initialise act - from 0-2

action = [] # initialise action vector - action = actions[act]

r = 0 # initiliase reward

state = [0,0,0,0] #starting state of the two particles. p1,v1,p2,v2

#w = numpy.zeros((len(actions),len(state)))

#w = np.array([[0.2,0.1,0,-1,-0.5,0.04,0.6,-0.32,0.11],[0.3,0.2,-1.007,0.06,-0.61,0.6,0.4,-0.8,0.07],[-0.3,0.1,-1.2,-0.87,-0.1,-0.4,-0.26,0.1,0.01],[1.4,-1.2,-0.45,0.08,-0.4,-0.1,-0.8,0.3]])

w = np.random.normal(0,0.1,(len(actions),len(state))) # initialise weights in the network

q1 = [0,0,0] # initialise q1 vector - predictor of the quality of a move

maxQ1 = 0 # initiliase maxQ 

targetQ = [0,0,0] # initialise target q

loss_delta = [] # loss delta - targetQ - q1

loss = [] # 1/2(loss_delta)**2

b = numpy.zeros((len(state),len(actions)),) # initialise back propogration matrix

gamma = 0.9 # discount on qs

reward_discount = 0.95

reward_counts = [] # initialise reward vector

j_counts = [] # how long did it last?

delta = 0 # delta between states

previous_delta = abs(state[0] - state[3]) # previous delta between states. to compare how the agents are going

target_delta = 0 # target distancebetween agents

delta_threshold = 4 # threshold distance between agents

e = 0.5 #0.94 # e-greedy exploration factor - this starts low and gets higher the longer the algorithm runs
# more exploration early, exploitation later

lr = 1 # learning rate

error = 0.1 # see 'get reward function'

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
    
    new_state = [0,0,state[0],state[1]]
    delta = 0
    
    #update velocities
    new_state[1] = state[1] + action[0]       
        
    #update positions
    new_state[0] = (state[0] + new_state[1]) %10    
    
    state = new_state
    
    delta = min(abs(state[0] - 0), abs((state[0]-10) - 0),abs((state[0]) - (-10)))
    
    #delta = abs(state[0] - state[2]) 
              
    return state, delta
    

def get_reward(delta,previous_delta, state):
    
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
        
    if  abs(delta) <= 1 and abs(state[1]) <= 0.21:
        r = 0.5
        if abs(delta) <= 0.3:
            r = 2
            if abs(delta) <= 0.1 and abs(state[1]) <= 0.11:
                r = 10
        #if abs(state[1]) < 0.2: 
         #   r = 10
    #elif delta < previous_delta:
     #   r = 0.1
    else:
        r = -0.05 #this is only if time matters, otherwise should it be punished?             
           
    return r

def back_propogate(state,loss_delta,w):
    
    b = numpy.zeros((len(state),len(actions)),) # backprop matrix

    #print (loss_delta)         
                        
    b = np.outer(loss_delta,state) # 6*9 matrix dl/dw = dl/dq * dq/dw - you should get this to work as the other method is computational inefficient!
    
    #print(w)
    #print()
        
    b = lr*b
    
    print (b)
    #print('next')
    #print(w)
                        
    w = np.add(b,w) # add new weights times learning rate WHAT IS GOING ON HERE!!!???? - subtract works fine!!!
    print (w)   
    

    #print(np.subtract(w,lr*b))        
        
    return w

def experience_replay(q,q1,r):
    
    maxQ1 = numpy.max(q1) # set max q1 - target value
                
    targetQ = q1 # ready the target vector
    
    print (q)
    #print ()
    print (q1)
    #print()
    print (maxQ1)
    #print()
    print(r)
    #print()
        
    targetQ[0:9] = (r + gamma*maxQ1) # set the target vector
    
    print (targetQ)
    #print()         
            
    loss_delta = np.subtract(targetQ,q) # delta Q
    
    act = np.argmax(q)
    
    if r > 0:
        loss_delta[act] += max(loss_delta) + 0.1  
        
    print (loss_delta)
    #print()
    
    #print (loss_delta)
                       
    #loss = (1/2)*(loss_delta)**2 # calculate loss vector   
       
    return loss_delta

"""Run training"""

for c in range(1000):

    j = 0 # counter
    
    totalReward = 0 # reward counter
    
    state = np.array([2,0,2,0]) # [current position, current velocity, previous position, previous velocity]
    
    memories = [] # reset memories
    
    rewards = [] # for discounting
    
    discounted_rewards = [] # discounted rewards
    
    previous_delta = 2 # reset
    
    delta = 0 # reset
    
    print(w)
    
    while j <= 50: # run until the points have moved away from one another
    #for t in range(25):
           
        r = 0 # reset reward
        
        q = w.dot(state) #re-calc q values
        
        q = softmax(q)    
        
        print (state)
        #print()
        #print (w)
        #print()
        #print(q)
        #print()
        if np.random.rand(1) < e: # e-greedy action selection
            act = np.argmax(q)
        else:
            act = np.random.randint(0,2)
            #print('explore!')
            q[act] = 1            
        
        action = actions[act] # action to take
        
        #print(act)
        #print()        
        #print (action)
        #print ()
        
        stateOld = state # keep old state for memory        
                
        state, delta =  update_state(state,action) # update state and calcualte delta       
                
        r = get_reward(delta,previous_delta, state) # get reward
        
        #print (r)
        #print(state)
        
        previous_delta = delta # update previous delta for next iteration
                
        q1 = w.dot(state) # get new predicted q values from new state

        q1 = softmax(q1)                  
                        
        memories.append([stateOld,q1,r,state,q]) # record memories for experience replay               

        rewards.append(r)  
                        
        if j % 25 == 0 and j != 0: # experience replay every 25 steps
            print ('hey')
            for b in range(25):                                                
                
                #i = random.randint(0,len(memories) -1)                              
                
                batch = memories[b] # retrieve memory               
                                                
                discounted_rewards = discount_rewards(rewards) # apply discount to the rewards vector                                                        
                
                loss_delta = experience_replay(batch[4],batch[1],discounted_rewards[b]) # calculate loss vector for memory              
                                                   
                w = back_propogate(batch[0],loss_delta,w) # back prop using memory (s0,loss,weights)                
                
            memories = [] # reset memories
            
                
        """
        elif j< 50: # for some early immediate changes
            
            loss_delta = experience_replay(q,q1,r)
            
            w = back_propogate(stateOld,loss_delta,w)
                
        loss_delta = experience_replay(q,q1,r)
            
        w = back_propogate(stateOld,loss_delta,w)
        
        """
                                
        totalReward += r
            
        j+= 1
        
        if e < 0.75:
        
            e += 0.01
        
        #print (state)
        #print(r)
      
    print(rewards)
    print (discounted_rewards)
    print (totalReward)
    print(j)

    reward_counts.append((totalReward))
    #j_counts.append(j)
          
       
        
    
plt.plot(reward_counts)
plt.show()

#plt.plot(j_counts)
#plt.show()

    
    
    
    