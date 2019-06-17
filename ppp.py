from __future__ import print_function, division
import multiprocessing
import os
import sys
import gc
import numpy as np
import random
import time
import operator
import math
import matplotlib as mpl
mpl.use('agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import multivariate_normal
from scipy.stats import norm
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
class ptReplica(multiprocessing.Process):

    def __init__(self, ep_no,cartpole, use_langevin_gradients, learn_rate,  w,  minlim_param, maxlim_param, samples, topology, burn_in, temperature, swap_interval, langevin_prob, path, parameter_queue, main_process,event ):
        #MULTIPROCESSING VARIABLES
        multiprocessing.Process.__init__(self)
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event =  event
        self.cartpole=cartpole
        self.temperature = temperature
        self.ep_no = ep_no
        self.adapttemp = temperature
        self.swap_interval = swap_interval
        self.path = path
        self.burn_in = burn_in
        #FNN CHAIN VARIABLES (MCMC)
        self.samples = samples
        self.topology = topology
        # self.train_x = trainx
        # self.train_y = trainy
        # self.test_x = testx
        # self.test_y = testy
        self.w = w
        self.minY = np.zeros((1,1))
        self.maxY = np.zeros((1,1))

        self.minlim_param = minlim_param
        self.maxlim_param = maxlim_param


        self.use_langevin_gradients = use_langevin_gradients

        self.sgd_depth = 1 # always should be 1

        self.learn_rate = learn_rate

        self.l_prob = langevin_prob  # can be evaluated for diff problems - if data too large keep this low value since the gradients cost comp time


    def rmse(self, array):
        return np.sqrt(((array) ** 2).mean())

    #def rmse(self, pred, actual): 

     #   return np.sqrt(((pred-actual)**2).mean())


    '''def likelihood_func(self, fnn, data, w):
        y = data[:, self.topology[0]]
        fx  = fnn.evaluate_proposal(data,w)
        rmse = self.rmse(fx,y)
        z = np.zeros((data.shape[0],self.topology[2]))
        lhood = 0
        for i in range(data.shape[0]):
            for j in range(self.topology[2]):
                if j == y[i]:
                    z[i,j] = 1
                lhood += z[i,j]*np.log(prob[i,j])
        return [lhood/self.temperature, fx, rmse]'''


    def likelihood_func(self, cartpole, ep_no, w, tausq):
        fx = cartpole.run_single_episode(ep_no, w)
       # y = y#data[:, self.topology[0]]
        #fx = rnn.evaluate_proposal(x,w)
        #rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 *(np.square(cartpole.loss()))  / tausq
        return [np.sum(loss)/self.adapttemp, fx, self.rmse(cartpole.loss())]
        #loss = np.sum(-0.5*np.log(2*math.pi*tau_sq) - 0.5*np.square(y-fx)/tau_sq)
        #return [np.sum(loss)/self.adapttemp, fx, rmse]

    '''def prior_likelihood(self, sigma_squared, nu_1, nu_2, w):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + self.topology[2]+h*self.topology[2]) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2
        return log_loss'''

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def run(self):
        #INITIALISING FOR RNN
        #testsize = len(self.test_x)
        #trainsize = len(self.train_x)
        samples = self.samples 
        
        
        #x_test = self.test_x
        #x_train = self.train_x
        netw = self.topology
        #y_test = self.test_y
        #y_train = self.train_y

        batch_save = 10  # batch to append to file


        
        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2] + (netw[1] * netw[1]) # num of weights and bias
        pos_w = np.ones((samples, w_size)) #Posterior for all weights
        #pos_w = np.ones((samples, w_size)) #Posterior for all weights
        lhood_list = np.zeros((samples,1))
        surrogate_list = np.zeros((samples,1))
        #fxtrain_samples = np.ones((batch_save, trainsize)) #Output of regression FNN for training samples
        #fxtest_samples = np.ones((batch_save, testsize)) #Output of regression FNN for testing samples
        rmse_train  = np.zeros(samples)
        rmse_test = np.zeros(samples)
        acc_train = np.zeros(samples)
        acc_test = np.zeros(samples)
        learn_rate = self.learn_rate

        #Random Initialisation of weights
        w = self.w
        eta = 0 #Junk variable 
        #print(w,self.temperature)
        w_proposal = np.random.randn(w_size)
        #Randomwalk Steps
        step_w = 0.025

        step_eta = 0.2
        #Declare RNN
       # cartpole = cp.Network(learn_rate, self.topology)

        #print(self.topology, ' topo')
        #Evaluate Proposals
        #pred_train  = rnn.evaluate_proposal(x_train,w) #    
        #pred_test  = rnn.evaluate_proposal(x_test, w) #
        #Check Variance of Proposal

        eta = np.log(np.var(self.cartpole.loss()))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
        np.fill_diagonal(sigma_diagmat, step_w)

        delta_likelihood = 0.5 # an arbitrary position 
        prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

 

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(self.cartpole,-1, w, tau_pro)
        [_, pred_test, rmsetest] = self.likelihood_func(self.cartpole,-1, w, tau_pro)


        trainacc = 0
        testacc=0

        prop_list = np.zeros((samples,w_proposal.size))
        likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
        likeh_list[0,:] = [-100, -100] # to avoid prob in calc of 5th and 95th percentile later
        surg_likeh_list = np.zeros((samples,2))
        accept_list = np.zeros(samples)

        num_accepted = 0

        langevin_count = 0

        pt_samples = samples * 0.6 # this means that PT in canonical form with adaptive temp will work till pt  samples are reached

        init_count = 0
        for i in range(samples-1):  # Begin sampling --------------------------------------------------------------------------

            timer1 = time.time() 

            if i < pt_samples:
                self.adapttemp =  self.temperature #* ratio  #

            if i == pt_samples and init_count ==0: # move to MCMC canonical
                self.adapttemp = 1
                [likelihood, pred_train, rmsetrain ] = self.likelihood_func(self.cartpole, i, w, tau_pro)
                [_, pred_test, rmsetest ] = self.likelihood_func(self.cartpole,i, w, tau_pro)
                init_count = 1


            lx = np.random.uniform(0,1,1)

            if (self.use_langevin_gradients is True) and (lx< self.l_prob):  
                # w_gd = rnn.langevin_gradient(self.train_x,self.train_y, w.copy()) # Eq 8
                # w_proposal = np.random.normal(w_gd, step_w, w_size) # Eq 7
                # w_prop_gd = rnn.langevin_gradient(self.train_x,self.train_y, w_proposal.copy()) 
                # #first = np.log(multivariate_normal.pdf(w , w_prop_gd , sigma_diagmat)) 
                # #second = np.log(multivariate_normal.pdf(w_proposal , w_gd , sigma_diagmat)) # this gives numerical instability - hence we give a simple implementation next that takes out log 

                # wc_delta = (w- w_prop_gd) 
                # wp_delta = (w_proposal - w_gd )

                # sigma_sq = step_w

                # first = -0.5 * np.sum(wc_delta  *  wc_delta  ) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                # second = -0.5 * np.sum(wp_delta * wp_delta ) / sigma_sq

            
                # diff_prop =  first - second 
                # diff_prop =  diff_prop/self.adapttemp  
                # langevin_count = langevin_count + 1
                pass
                

            else:
                diff_prop = 0
                w_proposal = np.random.normal(w, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

  

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(self.cartpole, i, w_proposal,tau_pro) 

            [_, pred_test, rmsetest] = self.likelihood_func(self.cartpole, i, w_proposal,tau_pro)
            
            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,tau_pro)  # takes care of the gradients
            diff_prior = prior_prop - prior_current
            diff_likelihood = likelihood_proposal - likelihood


            surg_likeh_list[i+1,0] = likelihood_proposal * self.adapttemp
            #surg_likeh_list[i+1,1] = np.nan  

            try:
                mh_prob = min(0, (diff_likelihood+diff_prior+ diff_prop))
                mh_prob = math.exp(mh_prob)
            except OverflowError as e:
                mh_prob = 1



            accept_list[i+1] = num_accepted

            if (i % batch_save+1) == 0: # just for saving posterior to file - work on this later
                x = 0



            u = random.uniform(0, 1)

            
            prop_list[i+1,] = w_proposal    
            likeh_list[i+1,0] = likelihood_proposal



            if u < mh_prob:
                num_accepted  =  num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = w_proposal 

                eta = eta_pro

                acc_train[i+1,] = 0
                acc_test[i+1,] = 0

                #print (i, langevin_count, self.adapttemp, diff_prop ,  likelihood, rmsetrain, rmsetest, acc_train[i+1,], acc_test[i+1,] , 'accepted') 
                print(i,rmsetrain,rmsetest,'accepted')
                pos_w[i+ 1,] = w_proposal

                #fxtrain_samples[i + 1,] = pred_train
                #fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest 
                #x = x + 1

            else:
                pos_w[i+1,] = pos_w[i,] 
                #fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                #fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]
                acc_train[i+1,] = acc_train[i,]
                acc_test[i+1,] = acc_test[i,]

                

            if (i % self.swap_interval == 0 and i != 0 ):
                print('\nTemperature: {} Swapping weights: {}'.format(self.temperature, w[:2]))
                param = np.concatenate([w, np.asarray([eta]).reshape(1), np.asarray([likelihood*self.temperature]),np.asarray([self.temperature])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.wait() 
                if not self.parameter_queue.empty() :
                    try:
                        result =  self.parameter_queue.get()
                        w = result[0:self.w_size]
                        eta = result[self.w_size]
                        likelihood = result[self.w_size+1]/self.temperature
                        print('Temperature: {} Swapped weights: {}'.format(self.temperature, w[:2]))
                    except:
                        print ('error')
                #else:
                    #print("Khali - swap bug fix by Arpit Kapoor) ")
                self.event.clear()

            #elapsed_time = ":".join(Replica.convert_time(time.time() - self.start_time)) 


        param = np.concatenate([w, np.asarray([eta]).reshape(1), np.asarray([likelihood]),np.asarray([self.adapttemp]),np.asarray([i])])

        self.parameter_queue.put(param) 

        print ((num_accepted*100 / (samples * 1.0)), '% was accepted')
        accept_ratio = num_accepted / (samples * 1.0) * 100 


        print ((langevin_count*100 / (samples * 1.0)), '% was Lsnngrevin ')
        langevin_ratio = langevin_count / (samples * 1.0) * 100 

        
        file_name = self.path+'/posterior/pos_w/'+'chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,pos_w )
        
        #file_name = self.path+'/predictions/fxtrain_samples_chain_'+ str(self.temperature)+ '.txt'
        #np.savetxt(file_name, fxtrain_samples, fmt='%1.2f')
        #file_name = self.path+'/predictions/fxtest_samples_chain_'+ str(self.temperature)+ '.txt'
        #np.savetxt(file_name, fxtest_samples, fmt='%1.2f')        
        file_name = self.path+'/predictions/rmse_test_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, rmse_test, fmt='%1.8f')        
        file_name = self.path+'/predictions/rmse_train_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, rmse_train, fmt='%1.8f')


        file_name = self.path+'/predictions/acc_test_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, acc_test, fmt='%1.2f')        
        file_name = self.path+'/predictions/acc_train_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, acc_train, fmt='%1.2f')



        file_name = self.path+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,likeh_list, fmt='%1.4f')  

        file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
        np.savetxt(file_name, [accept_ratio], fmt='%1.4f')

        file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, accept_list, fmt='%1.4f')



        self.signal_main.set() 

class ParallelTempering:

    def __init__(self,ep_no,  use_langevin_gradients, learn_rate,  topology, num_chains, maxtemp, NumSample, swap_interval, langevin_prob, path):
        #FNN Chain variables
        # self.train_x = train_x
        # self.train_y = train_y
        # self.test_x = test_x
        # self.test_y = test_y
        self.topology = topology
        self.num_param = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2] + (topology[1] * topology[1])
        #Parallel Tempering variables
        self.swap_interval = swap_interval
        self.path = path
        self.maxtemp = maxtemp
        self.ep_no=ep_no
        self.langevin_prob = langevin_prob
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = int(NumSample/self.num_chains)
        self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
        # create queues for transfer of parameters between process chain
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()    
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        
        self.all_param = None
        self.geometric = True # True (geometric)  False (Linear)

        self.minlim_param = 0.0
        self.maxlim_param = 0.0
        self.minY = np.zeros((1,1))
        self.maxY = np.ones((1,1))

        self.model_signature = 0.0
        self.learn_rate = learn_rate
        self.use_langevin_gradients = use_langevin_gradients
        self.cartpole = cp.CartPole(self.topology,self.NumSamples,self.learn_rate)


    def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        
        """

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                        2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                        2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                        1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                        1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                        1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                        1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                        1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                        1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                        1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                        1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                        1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                        1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                        1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                        1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                        1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                        1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                        1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                        1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                        1.26579, 1.26424, 1.26271, 1.26121,
                        1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
        else:
            tstep = tstep[ndim-1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas
        
    def assign_temperatures(self):
        # #Linear Spacing
        # temp = 2
        # for i in range(0,self.num_chains):
        #     self.temperatures.append(temp)
        #     temp += 2.5 #(self.maxtemp/self.num_chains)
        #     print (self.temperatures[i])
        #Geometric Spacing

        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)      
            for i in range(0, self.num_chains):         
                self.temperatures.append(np.inf if betas[i] is 0 else 1.0/betas[i])
                #print (self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp /self.num_chains)
            temp = 1
            for i in range(0, self.num_chains):            
                self.temperatures.append(temp)
                temp += tmpr_rate
                #print(self.temperatures[i])


    def initialize_chains(self,  burn_in):
        self.burn_in = burn_in
        self.assign_temperatures()
        self.minlim_param = np.repeat([-100] , self.num_param)  # priors for nn weights
        self.maxlim_param = np.repeat([100] , self.num_param)


        
        for i in range(0, self.num_chains):

            w = np.random.randn(self.num_param)
            self.chains.append(ptReplica(self.ep_no,self.cartpole, self.use_langevin_gradients, self.learn_rate, w,  self.minlim_param, self.maxlim_param, self.NumSamples,self.topology,self.burn_in,self.temperatures[i],self.swap_interval, self.langevin_prob, self.path,self.parameter_queue[i],self.wait_chain[i],self.event[i]))

    def surr_procedure(self,queue):

        if queue.empty() is False:
            return queue.get()
        else:
            return
    
    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
            param1 = parameter_queue_1.get()
            param2 = parameter_queue_2.get()
            w1 = param1[0:self.num_param]
            eta1 = param1[self.num_param]
            lhood1 = param1[self.num_param+1]
            T1 = param1[self.num_param+2]
            w2 = param2[0:self.num_param]
            eta2 = param2[self.num_param]
            lhood2 = param2[self.num_param+1]
            T2 = param2[self.num_param+2]
            #print('yo')
            #SWAPPING PROBABILITIES
            try:
                swap_proposal =  min(1,0.5*np.exp(min(709, lhood2 - lhood1)))
            except OverflowError:
                swap_proposal = 1
            u = np.random.uniform(0,1)
            if u < swap_proposal: 
                self.total_swap_proposals += 1
                self.num_swap += 1
                param_temp =  param1
                param1 = param2
                param2 = param_temp
            return param1, param2
        else:
            self.total_swap_proposals += 1
            return


    def run_chains(self): 
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.num_chains-1) 
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))  
        lhood = np.zeros(self.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples-1
        number_exchange = np.zeros(self.num_chains)
        filen = open(self.path + '/num_exchange.txt', 'a')
        #RUN MCMC CHAINS
        for l in range(0,self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0,self.num_chains):        
            self.chains[j].start()
        #SWAP PROCEDURE

        while True:
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count+=1
                    print(str(self.chains[index].temperature) +" Dead"+str(index))

            if count == self.num_chains:
                break
            print("Waiting")
            timeout_count = 0
            for index in range(0,self.num_chains):
                print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait(timeout=5)
                if flag:
                    print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                print("Skipping the swap!")
                continue
            print("Event occured")
            for index in range(0,self.num_chains-1):
                print('starting swap')
                try:
                    param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
                    self.parameter_queue[index].put(param_1)
                    self.parameter_queue[index+1].put(param_2)
                    if index == 0:
                        if swapped:
                            swaps_appected_main += 1
                        total_swaps_main += 1
                except:
                    print("Nothing Returned by swap method!")
            for index in range (self.num_chains):
                    self.event[index].set()
                    self.wait_chain[index].clear()

        print("Joining processes")

        #JOIN THEM TO MAIN PROCESS
        for index in range(0,self.num_chains):
            self.chains[index].join()
        self.chain_queue.join()
        
        

        pos_w,  rmse_train, rmse_test, acc_train, acc_test,  likelihood_vec ,   accept_vec, accept  = self.show_results()




        print("NUMBER OF SWAPS =", self.num_swap)
        swap_perc = self.num_swap*100/self.total_swap_proposals  

        return pos_w,  rmse_train, rmse_test, acc_train, acc_test,   likelihood_vec , swap_perc,    accept_vec, accept



    def show_results(self):

        burnin = int(self.NumSamples*self.burn_in)

        likelihood_rep = np.zeros((self.num_chains, self.NumSamples - 1, 2)) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        accept_percent = np.zeros((self.num_chains, 1))
        accept_list = np.zeros((self.num_chains, self.NumSamples )) 

        pos_w = np.zeros((self.num_chains,self.NumSamples - burnin, self.num_param)) 

        #fx_train_all  = np.zeros((self.num_chains,self.NumSamples - burnin, self.train_x.shape[0]))
        rmse_train = np.zeros((self.num_chains,self.NumSamples - burnin))
        acc_train = np.zeros((self.num_chains,self.NumSamples - burnin))
       # fx_test_all  = np.zeros((self.num_chains,self.NumSamples - burnin, self.test_x.shape[0]))
        rmse_test = np.zeros((self.num_chains,self.NumSamples - burnin))
        acc_test = np.zeros((self.num_chains,self.NumSamples - burnin))

        
        
        for i in range(self.num_chains):
            file_name = self.path+'/posterior/pos_w/'+'chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            pos_w[i,:,:] = dat[burnin:,:]  

            file_name = self.path + '/posterior/pos_likelihood/'+'chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            likelihood_rep[i, :] = dat[1:]


            file_name = self.path + '/posterior/accept_list/' + 'chain_'  + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            accept_list[i, :] = dat 


            #file_name = self.path+'/predictions/fxtrain_samples_chain_'+ str(self.temperatures[i])+ '.txt'
            #dat = np.loadtxt(file_name)
            #fx_train_all[i,:,:] = dat[burnin:,:]

            #file_name = self.path+'/predictions/fxtest_samples_chain_'+ str(self.temperatures[i])+ '.txt'
            #dat = np.loadtxt(file_name)
            #fx_test_all[i,:,:] = dat[burnin:,:]    

            file_name = self.path+'/predictions/rmse_test_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            rmse_test[i,:] = dat[burnin:]    

            file_name = self.path+'/predictions/rmse_train_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            rmse_train[i,:] = dat[burnin:]

            file_name = self.path+'/predictions/acc_test_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            acc_test[i,:] = dat[burnin:]    

            file_name = self.path+'/predictions/acc_train_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            acc_train[i,:] = dat[burnin:]


        posterior = pos_w.transpose(2,0,1).reshape(self.num_param,-1)  
        
        
        #fx_train = fx_train_all.transpose(2,0,1).reshape(self.train_x.shape[0],-1)  # need to comment this if need to save memory 
        #fx_test = fx_test_all.transpose(2,0,1).reshape(self.test_x.shape[0],-1) 

        #fx_test = fxtest_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.testdata.shape[0]) # konarks version


        likelihood_vec = likelihood_rep.transpose(2,0,1).reshape(2,-1) 

        rmse_train = rmse_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        acc_train = acc_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        rmse_test = rmse_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        acc_test = acc_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)


        accept_vec  = accept_list 

        #print(accept_vec)







        accept = np.sum(accept_percent)/self.num_chains 

        #np.savetxt(self.path + '/pos_param.txt', posterior.T)  # tcoment to save space
        
        np.savetxt(self.path + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

        np.savetxt(self.path + '/accept_list.txt', accept_list, fmt='%1.2f')

        np.savetxt(self.path + '/acceptpercent.txt', [accept], fmt='%1.2f')


        return posterior,   rmse_train, rmse_test,  acc_train, acc_test,  likelihood_vec.T, accept_vec , accept

    def make_directory (self, directory): 
        if not os.path.exists(directory):
            os.makedirs(directory)

def main():
    #topology
    hidden = 5
    input = 4  
    output =2
    topology = [input, hidden, output]
    numSamples = 5000  # need to decide yourself
    #learnRate = 0.01
    maxtemp = 10
 
    num_chains = 10
    swap_ratio = 0.05
    swap_interval = int(swap_ratio * numSamples/num_chains)    # int(swap_ratio * (NumSample/num_chains)) #how ofen you swap neighbours. note if swap is more than Num_samples, its off
    burn_in = 0.5
     
    learn_rate = 0.01  # in case langevin gradients are used. Can select other values, we found small value is ok. 

    use_langevin_gradients =False
    problemfolder = os.getcwd()+'/Res_LG-Lprob/'  #'/home/rohit/Desktop/PT/Res_LG-Lprob/'  # change this to your directory for results output - produces large datasets

    problemfolder_db = 'Res_LG-Lprob/'  # save main results




    filename = ""
    run_nb = 0
    name="cartpole"
    while os.path.exists( problemfolder+name+'_%s' % (run_nb)):
        run_nb += 1
    if not os.path.exists( problemfolder+name+'_%s' % (run_nb)):
        os.makedirs(  problemfolder+name+'_%s' % (run_nb))
        path = (problemfolder+ name+'_%s' % (run_nb))

    filename = ""
    run_nb = 0
    while os.path.exists( problemfolder_db+name+'_%s' % (run_nb)):
        run_nb += 1
    if not os.path.exists( problemfolder_db+name+'_%s' % (run_nb)):
        os.makedirs(  problemfolder_db+name+'_%s' % (run_nb))
        path_db = (problemfolder_db+ name+'_%s' % (run_nb))



    resultingfile = open( path+'/master_result_file.txt','a+') 

    resultingfile_db = open( path_db+'/master_result_file.txt','a+')  

    timer = time.time() 

    langevin_prob = 1/10
    


    pt = ParallelTempering(-1,use_langevin_gradients,  learn_rate, topology, num_chains, maxtemp, numSamples, swap_interval, langevin_prob, path)

    directories = [  path+'/predictions/', path+'/posterior', path+'/results', path+'/surrogate', path+'/surrogate/learnsurrogate_data', path+'/posterior/pos_w',  path+'/posterior/pos_likelihood',path+'/posterior/surg_likelihood',path+'/posterior/accept_list'  ]

    for d in directories:
        pt.make_directory((filename)+ d)    



    pt.initialize_chains(  burn_in)

    
    pos_w,  rmse_train, rmse_test, acc_train, acc_test,   likelihood_rep , swap_perc,    accept_vec, accept = pt.run_chains()

    list_end = accept_vec.shape[1] 
    accept_ratio = accept_vec[:,  list_end-1:list_end]/list_end   
    accept_per = np.mean(accept_ratio) * 100

    print(accept_per, ' accept_per')





    timer2 = time.time()

    timetotal = (timer2 - timer) /60
    print ((timetotal), 'min taken')


    


if __name__ == "__main__": main()
























