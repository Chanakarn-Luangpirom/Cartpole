import torch
from torch import nn
from utils import make_network
import numpy as np

class QNetwork(nn.Module):
    def __init__(self,
                 gamma,
                 state_dim,
                 action_dim,
                 hidden_sizes=[10, 10]):
        super().__init__()
        self.gamma = gamma
        
        # neural net architecture
        self.network = make_network(state_dim, action_dim, hidden_sizes)
        
        self.action_dim = action_dim
    
    def forward(self, states):
        '''Returns the Q values for each action at each state.'''
        qs = self.network(states)
        return qs

    def get_max_q(self, states):
        # TODO: Get the maximum Q values of all states s.
        q , _ = torch.max(self.forward(states).data,dim = 1)
        return q
        

    
    def get_action(self, state, eps):
        # TODO: Get the action at a given state according to an epsilon greedy method.
        if np.random.rand() <= eps:
            # Take random action
            return np.array(np.random.randint(self.action_dim))
        else:
            # Current Optimal
            return np.array(np.argmax(self.forward(state)))


    
    @torch.no_grad()
    def get_targets(self, rewards, next_states, dones):
        
        """
         Modify the get targets function in network.py, which, given rewards r, next states s
        ′ and terminal signals d, computes the target for our Q function. See the forward method for additional info.
        Note that if s is a terminal state (given by the done flag), the Q function should return 0 by definition. All
        inputs are PyTorch tensors, and the rewards are of shape B, the next states are of shape (B, state_dim), and the
        terminal signals are of shape B, where B is the batch size. Your function should return a tensor of size B.

        """
        ##1 means not done / 0 means done
        rhs = self.gamma*self.get_max_q(next_states)
        lhs = rewards.flatten()
        dones = dones.flatten()
        dones_new = 1-dones
        target = torch.add(rhs, lhs)
        target = torch.multiply(target,dones_new)
        return target

          