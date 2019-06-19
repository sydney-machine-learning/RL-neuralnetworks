import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy 
import cartpole as cp

#Hyperparameters
NUM_EPISODES = 10000
LEARNING_RATE = 0.000025
GAMMA = 0.99


class MCMC:
    def __init__(self, samples, lrate , topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.lrate = lrate
        # ----------------
        self.NUM_EPISODES = samples #10000
        self.LEARNING_RATE = lrate #0.000025
        self.GAMMA = 0.99
        self.Topo = topology
        # Create gym and seed numpy
        #self.env = gym.make('CartPole-v0')
        #self.nA = self.env.action_space.n
        #np.random.seed(1)
        #self.neuralnet = Network(Topo)
        # Init weight
        #w = np.random.rand(4, 2)
        self.w = np.random.randn(self.topology[0]*topology[1]+topology[1]*topology[2]+topology[1]+topology[2],1)
        # Keep stats for final print of graph
        self.episode_rewards = []
        self.cartpole = cp.CartPole(self.topology,self.samples,self.lrate)

    def rmse(self, array):
        return np.sqrt(((array) ** 2).mean())

    def likelihood_func(self, cartpole, ep_no, w, tausq):
        #fx = cartpole.run_single_episode(ep_no, w)
         if tausq==0:
             tausq=0.0001
        #rmse = self.rmse(fx, y)
       # loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
         loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 *(np.square(cartpole.actual()[ep_no]-cartpole.predicted()[ep_no]))  / tausq
         print(np.square(cartpole.actual()[ep_no]-cartpole.predicted()[ep_no]))
         return [loss, self.rmse(cartpole.loss())]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        if tausq==0:
            tausq=0.0001
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self):

        #testsize = len(self.test_x)   #self.testdata.shape[0]
        #trainsize = len(self.train_x)
        samples = self.samples
        #x_test = np.linspace(0, 1, num=testsize)
        #x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        ##y_test = self.test_y  #self.testdata[:, netw[0]]
        #y_train = self.train_y #self.traindata[:, netw[0]]
        #print(len(y_train))
        #print(len(y_test))

        # here
        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2] # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        # original -->    fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        #print('shape: ',np.array(y_train).shape[1])
        #fxtrain_samples = np.ones((samples, trainsize,int(np.array(y_train).shape[1])))  # fx of train data over all samples
        # original --> fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples || probably for 1 dimensional data
        #fxtest_samples = np.ones((samples, testsize,np.array(self.test_y).shape[1]))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        step_w = 0.09  # defines how much variation you need in changes to w
        step_eta = 0.07
        # --------------------- Declare FNN and initialize

        #self.cartpole = cp.CartPole(self.learnrate,self.topology, self.train_x,self.train_y,self.test_x,self.test_y)
        print ('evaluate Initial w')
        #print(w,np.array(self.train_x).shape)
        self.cartpole.run_single_episode(0, w)
        #pred_test = neuralnet.evaluate_proposal(self.test_x, w)

        #eta = np.log(np.var(np.array(pred_train) - np.array(y_train)))
        
        eta = np.log(np.var((np.square(self.cartpole.actual()-self.cartpole.predicted()))))
        tau_pro = np.exp(eta)
        #err_nn = np.sum(np.square(np.array(pred_train) - np.array(y_train)))/(len(pred_train)) #added by ashray mean square sum
        #print('err_nn is: ',err_nn)
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        #print(pred_train)
        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients
        [likelihood, rmsetrain] = self.likelihood_func(self.cartpole, 0, w, tau_pro)
        #[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(self.cartpole, self.test_x,self.test_y, w, tau_pro)

        print(likelihood,' is likelihood of train')
        #print(pred_train)
        #print(pred_train, ' is pred_train')
        naccept = 0
        print ('begin sampling using mcmc random walk')
        #plt.plot(x_train, y_train)
        #plt.plot(x_train, pred_train)
        #plt.title("Plot of Data vs Initial Fx")
        #plt.savefig('mcmcresults/begin.png')
        #plt.clf()

       #plt.plot(x_train, y_train)

        for i in range(samples-1):
            #print(i)
            self.cartpole.run_single_episode(0, w)
            for j in range(0,len(self.cartpole.getrewards())):

                w_proposal = w + np.random.normal(0, step_w, w_size)

                eta_pro = eta + np.random.normal(0, step_eta, 1)
                tau_pro = math.exp(eta_pro)

                [likelihood_proposal, rmsetrain] = self.likelihood_func(self.cartpole,j , w_proposal,
                                                                                tau_pro)
            
            #[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x,self.test_y, w_proposal,
            #                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

                prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

                diff_likelihood = likelihood_proposal - likelihood
                diff_priorliklihood = prior_prop - prior_likelihood

            #mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))
                mh_prob = min(0, (diff_likelihood + diff_priorliklihood))
                mh_prob = math.exp(mh_prob)
                u = random.uniform(0, 1)
            #print(rmsetrain)
            #quit()
                if u < mh_prob:
                # Update position
                #print(i, ' is the accepted sample')
                    naccept += 1
                    likelihood = likelihood_proposal
                    prior_likelihood = prior_prop
                    w = w_proposal
                    eta = eta_pro
                # if i % 100 == 0:
                #     #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                #     print ('Sample:',i, 'RMSE train:', rmsetrain, 'RMSE test:',rmsetest)

                    pos_w[i + 1,] = w_proposal
                    pos_tau[i + 1,] = tau_pro
                #fxtrain_samples[i + 1,] = pred_train
                #fxtest_samples[i + 1,] = pred_test
                    rmse_train[i + 1,] = rmsetrain
                #rmse_test[i + 1,] = rmsetest

               #plt.plot(x_train, pred_train)


                else:
                    pos_w[i + 1,] = pos_w[i,]
                    pos_tau[i + 1,] = pos_tau[i,]
                #fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                #fxtest_samples[i + 1,] = fxtest_samples[i,]
                    rmse_train[i + 1,] = rmse_train[i,]
                #rmse_test[i + 1,] = rmse_test[i,]

                # print i, 'rejected and retained'

                # if i % 100 == 0:
                # #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                #     print ('Sample:',i, 'RMSE train:', rmsetrain)

        print (naccept, ' num accepted')
        print ((naccept*100) / (samples * 1.0), '% was accepted')
        accept_ratio = naccept / (samples * 1.0) * 100

       #plt.title("Plot of Accepted Proposals")
       #plt.savefig('mcmcresults/proposals.png')
       #plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
       #plt.clf()

    #return #(pos_w, pos_tau, rmse_train, rmse_test, accept_ratio)


def main():
    #topology
    hidden = 5
    input = 4  
    output =2
    topology = [input, hidden, output]
    numSamples = 10000  # need to decide yourself
    learnRate = 0.02
    mcmc = MCMC(numSamples,learnRate, topology)  # declare class

    #[pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
    mcmc.sampler()
    print ('sucessfully sampled')


if __name__ == "__main__": main()












