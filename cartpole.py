import gym
import numpy as np
import matplotlib.pyplot as plt
import copy 
# An example of a class
class Network:

    def __init__(self, Topo):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer
        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer
        self.lrate = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # def sampleEr(self, actualout):
    #     error = np.subtract(self.out, actualout)
    #     sqerror = np.sum(np.square(error)) / self.Top[2]
    #     return sqerror

    def ForwardPass(self, X,w):
        self.decode(w)
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer
        return np.copy(np.array(self.out))
    
    def BackwardPass(self, Input, desired, vanilla):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

    # def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

    #     self.decode(w)  # method to decode w into W1, W2, B1, B2.

    #     size = data.shape[0]

    #     Input = np.zeros((1, self.Top[0]))  # temp hold input
    #     Desired = np.zeros((1, self.Top[2]))
    #     fx = np.zeros(size)

    #     for pat in range(0, size):
    #         Input[:] = data[pat, 0:self.Top[0]]
    #         Desired[:] = data[pat, self.Top[0]:]

    #         self.ForwardPass(Input)
    #         fx[pat] = self.out

    #     return fx

class CartPole:

    def __init__(self,Topo,numsamples,learn_rate):
        #Hyperparameters
        self.NUM_EPISODES = numsamples #10000
        self.LEARNING_RATE = learn_rate #0.000025
        self.GAMMA = 0.99
        self.Topo = Topo
        # Create gym and seed numpy
        self.env = gym.make('CartPole-v0')
        self.nA = self.env.action_space.n
        #np.random.seed(1)
        self.neuralnet = Network(Topo)
        # Init weight
        #w = np.random.rand(4, 2)
        self.w = np.random.randn(Topo[0]*Topo[1]+Topo[1]*Topo[2]+Topo[1]+Topo[2],1)
        # Keep stats for final print of graph
        self.episode_rewards = []
        self.score = 0
        self.aprobs = []
        self.discountedsum = []



    # Our policy that maps state to action parameterized by w
    def policy(self,state,w):
        z = self.neuralnet.ForwardPass(state,w)
        #z = state.dot(w)
        exp = np.exp(z)
        return exp/np.sum(exp)

    # Vectorized softmax Jacobian
    def softmax_grad(self,softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    # Main loop 
    # Make sure you update your weights AFTER each episode
    #for e in range(1):
    def run_single_episode(self,ep_no,w):
        state = self.env.reset()[None,:]

        #grads = []	
        rewards = []

        # Keep track of game score to print
        self.score = 0
        self.aprobs= []
        while True:

            # Uncomment to see your model train in real time (slower)
            #env.render()

            # Sample from policy and take action in environment
            probs = self.policy(state,w)
            action = np.random.choice(self.nA,p=probs[0])
            next_state,reward,done,_ = self.env.step(action)
            next_state = next_state[None,:]
            #print(probs)
            action = 1 if np.random.uniform() < probs[0][1] else 0 # roll the dice!
            self.aprobs.append(action - probs[0][1])
            # # Compute gradient and save with reward in memory for our weight updates
            # dsoftmax = softmax_grad(probs)[action,:]
            # dlog = dsoftmax / probs[0,action]
            # grad = state.T.dot(dlog[None,:])

            # grads.append(grad)
            rewards.append(reward)		

            self.score+=reward

            # Dont forget to update your old state to the new state
            state = next_state

            if done:
                break
        
        self.discountedsum=[]        
        for i,item in enumerate(rewards):
            #print('hi ',self.discountedsum)
            self.discountedsum.append(sum([ r * (self.GAMMA ** t) for t,r in enumerate(rewards[i:])]))
        #print(self.discountedsum,'till terminal',len(rewards))
        self.discountedsum = np.array(self.discountedsum)
        self.discountedsum -= np.mean(self.discountedsum)
        self.discountedsum /= np.std(self.discountedsum)
        self.aprobs = np.array(self.aprobs)
        self.aprobs *= self.discountedsum
        #print(self.discountedsum,' asdafdadf')
        
        # Weight update
        # for i in range(len(grads)):

        #     # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
        #     w += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** t) for t,r in enumerate(rewards[i:])])

        
        # Append for logging and print
        self.episode_rewards.append(self.score) 
        print("EP: " + str(ep_no) + " Score: " + str(self.score) + "         ") 
    
    def sumrewards(self):
        return self.score

    def loss(self):
        return np.copy(self.aprobs)
    # plt.plot(np.arange(NUM_EPISODES),episode_rewards)
    # plt.show()
    # env.close()