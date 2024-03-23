import torch
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer(object):
    '''Replay buffer that stores online (s, a, r, s', d) transitions for training.'''
    def __init__(self, maxsize=100000):
        # TODO: Initialize the buffer using the given parameters.
        # HINT: Once the buffer is full, when adding new experience we should not care about very old data.
        self.buffer = deque([],maxsize)
#         Trans = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
#         self.trans = Trans
    
    def __len__(self):
        # TODO: Return the length of the buffer (i.e. the number of transitions).
        return len(self.buffer)
    
    def add_experience(self, state, action, reward, next_state, done):
        # TODO: Add (s, a, r, s', d) to the buffer.
        # HINT: See the transition data type defined at the top of the file for use here.
        if done == True:
            done = int(1)
        else:
            done = int(0)
        self.buffer.appendleft(Transition(state,action,reward,next_state,done))

                
    def sample(self, batch_size):
        # TODO: Sample 'batch_size' transitions from the buffer.
        # Return a tuple of torch tensors representing the states, actions, rewards, next states, and terminal signals.
        # HINT: Make sure the done signals are floats when you return them.
#         print('Buffer',self.buffer)
        samples = random.sample(self.buffer, batch_size)
        s_batch = []
        a_batch = []
        r_batch = []
        s_next_batch = []
        d_batch = []
        for sample in samples:
            s = sample[0]
            a = sample[1]
            r = sample[2]
            s_next = sample[3]
            d = sample[4]
            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)
            s_next_batch.append(s_next)
            d_batch.append(d)
            
           
        s_batch = torch.Tensor(torch.stack(s_batch))
        a_batch = torch.Tensor(torch.stack(a_batch))
        r_batch = torch.Tensor(torch.stack(r_batch))
        s_next_batch = torch.Tensor(torch.stack(s_next_batch))
        d_batch = torch.Tensor(d_batch)
        
        return (s_batch,a_batch,r_batch,s_next_batch,d_batch)
        
#         print(samples)
#         return samples
