# RL-neuralnetworks-
All files use gym openAI for the environment space.

The cartpole_ql file contains python code for solving the cartpole problem using Q-Learning and Neural networks implemented using keras.

The cartpole_mcmc.py uses random walks of a single monte carlo markov chain by adding random gaussian noises to our weights without using any gradient to solve the single cartpole problem. It accesses the cartpole environment from gym openAI and neural network class from cartpole.py

The ppp.py file infuses multiple mcmc chains at temperatures following a geometric progression. Each chain swaps weights with adjacent chains when sorted according to their respective temperatures. Infusing Parallel tempering increases the chances of getting a global optimum solution instead of a local optimum one. It also imports the environment(gym) and neural network class from cartpole.py


