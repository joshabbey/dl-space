% An n-by-m maze with a series of obstacles and some mines
% The agent has 60 moves to navigate to the goal
% A q learning agent with a neural network approximator is feed the state of the agent at time s
% This state is feed through the neural network and it predicts the quality of each action, q, using the agent NN
% The agent applies an e-greedy selection policy on these actions and chooses a
& The agent state in the maze is updated via the action, a, to give the new state s'
% The agent is given a reward, r, depending on the new state
% The agent predicts the value of each action, q', in the new state, s', using the target NN
% A tuple of [s,q,a,s',r,q'] is stored in the agents memory
% Discounted rewards are applied to all rewards in each game
% When learning, for each randomly chosen memory the 'loss' is equal to 1/2((q'(max)*discount) + r- q(chosenaction))
% This loss needs to then be backpropogated through the agent NN in order to learn
% No loss against 'actions' not used - example loss vector - [0,0,0,0.1]
% Every X training runs, the target network should be updated with the agent NN
% Learning is complete when the losses converge to zero
