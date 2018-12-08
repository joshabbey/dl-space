# An n-by-m maze with a series of obstacles and some mines
# The agent has 60 moves to navigate to the goal
# A q learning agent with a neural network approximator is feed the state of the agent at time s
# This state is feed through the neural network and it predicts the quality of each action, q, using the agent NN
# The agent applies an e-greedy selection policy on these actions and chooses a
# The agent state in the maze is updated via the action, a, to give the new state s'
# The agent is given a reward, r, depending on the new state
# The agent predicts the value of each action, q', in the new state, s', using the target NN
# A tuple of [s,q,a,s',r,q'] is stored in the agents memory
# Discounted rewards are applied to all rewards in each game
# When learning, for each randomly chosen memory the 'loss' is equal to 1/2((q'(max)*discount) + r- q(chosenaction))
# This loss needs to then be backpropogated through the agent NN in order to learn
# No loss against 'actions' not used - example loss vector - [0,0,0,0.1]
# Every X training runs, the target network should be updated with the agent NN
# Learning is complete when the losses converge to zero

# imports
import numpy as np

# Create maze class

class maze(self, int height, int width,int goal, int mines):
    def __init__(self, height, width, goal, mines):
        self.h = height
        self.w = width
        self.size = [height,width]
        self.maze = np.zeros((height,width))
        self.goal = goal
        self.mines = mines
    
    def updateMaze(self,int y, int x, str action):
        
        if action == 'up':
            if y =! 0 or self.maze[y -1][x] != 9:
                self.maze[y][x] =0
                self.maze[y-1][x] = 1
                
                y = y-1
        
        if action == 'right':
            if x =! self.w or self.maze[y][x+1] != 9:
                self.maze[y][x] =0
                self.maze[y][x+1] = 1
                
                x = x+1
                
        if action == 'down':
            if y =! self.h or self.maze[y + 1][x] != 9:
                self.maze[y][x] =0
                self.maze[y+1][x] = 1
                
                y = y+1
                
        if action == 'left':
            if x =! 0 or self.maze[y][x-1] != 9:
                self.maze[y][x] =0
                self.maze[y][x-1] = 1
                
                x = x-1
                
        return y,x
    
    def getReward(self,int state):
        if state == self.goal:
            return 100 # goal found
        else if state in self.mines:
            return -100 # landed on a mine
        else:
            return -0.1 # time penalty

class agent(self, str actions, length_input_layer, length_hidden_layer,length_output_layer, discount,epsilon,learning_rate,replace_target_iteration):
    
    # Create agent neural network
    def __init__(self):    
        self.input_layer = np.zeros((length_input_layer,1))
        self.weights1 = np.random.normal(0,1,(length_hidden_layer,length_input_layer))
        self.biases1 = np.random.normal(0,1,(1,length_hidden_layer))
        self.hidden_layer = np.zeros((1,length_hidden_layer))
        self.weights2 = np.random.normal(0,1,(length_output_layer,length_hidden_layer))
        self.biases2 = np.random.normal(0,1,(length_output_layer,1))
        self.output_layer = np.zeros((length_output_layer,1))
        self.actions = actions
        self.epsilon = episilon
        self.target_weights1 = weights1
        self.target_weights2 = weights2
        self.memories = []
        self.learning_rate = learning rate
        self.replace_target_iteration = replace_target_iteration
        
    def actionSelection(self):
        
        # e-greddy action selection policy
        if np.random.random_sample() >= self.epsilon:        
            return self.actions[argmax(self.output_layer)]
        else:
            return self.actions[np.random.randint(0,5)]
        
    def memoryStore(self,s,q,a,s1,r,q1):
        self.memories.append([s,q,a,s1,r,q1])
    
    def qualityDelta(self)
    
        
        
        
        
        
        
        
        
    
    
        
        
                
        
        
    