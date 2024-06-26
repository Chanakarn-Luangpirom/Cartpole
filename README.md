# Cartpole Balancing with DQN and LQR
This repository contains Python implementations of two different algorithms, Deep Q-Network (DQN) and Linear Quadratic Regulator (LQR), used to balance a cartpole system. These implementations were developed as part of a Reinforcement Learning class.

The cartpole balancing problem involves balancing a pole on a cart that can move along a frictionless track. The goal is to apply appropriate control inputs to the cart such that the pole remains upright. The two different approaches in the repository:

- **Deep Q-Network (DQN)**: A reinforcement learning algorithm that learns to balance the cartpole by approximating the optimal action-value function.
- **Linear Quadratic Regulator (LQR)**: A classical control method that computes optimal control inputs based on a linearized dynamics model of the cartpole system.

## Dependencies
To run the code in this repository, install the requirements from requirements.txt. Then, run the following. \
```python Cartpole_DQN/test.py```  or ```python Cartpole_LQR/test.py```



