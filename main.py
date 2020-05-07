#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 12:49:25 2018

@author: aidanrockea
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:32:53 2018

@author: aidanrocke & ildefonsmagrans
"""

## no net action!

#import random
import tensorflow as tf
import numpy as np

#import matplotlib.pyplot as plt
from agent import agent_cognition
from square_env import square_env
from utils import action_states
from visualisation import heatmap

## set random seed:
tf.set_random_seed(42)

# define training parameters:
horizon = 4
seed = 42
bound = 0.5
iters = 10000
batch_size = 50
lr = 0.01
prob = 0.8
R = 1.0

## define folder where things get saved:
folder = "/Users/aidanrockea/Desktop/vime/heat_maps/expt_7/"

export_dir = "/Users/aidanrockea/Desktop/vime/checkpoints/"

# define environment:
env = square_env(duration=horizon,radius=R,dimension=2*(horizon-1))

#tf.reset_default_graph()

A = agent_cognition(horizon,seed,bound)  

## define saver:
#saver = tf.train.Saver()

def main():
            
    with tf.Session() as sess:
        
        ### define beta schedule:
        betas = 1./np.array([min(0.001 + i/iters,1) for i in range(iters)])
        
        ## define inverse probability:
        inverse_prob = betas
        
        ### initialise the variables:
        sess.run(A.init_g)
                                
        for count in range(iters):
            
            ## reset the environment:
            env.reset()
            env.random_initialisation()
            
            mini_batch = np.zeros((batch_size*horizon,6))
            action_batch = np.zeros((batch_size*horizon,2))
            
            ## define mean and variance of environment:
            mu = env.dimension/2.0 - R ## mean of U(R,dimension-R)
            sigma = ((2*mu)**2)/12 ## variance of U(R,dimension-R)
            
            ### train our agent on a minibatch of recent experience:
            for i in range(batch_size):
                
                env.iter = 0
                                            
                if np.random.rand() > 1/inverse_prob[count]:
                    actions = A.random_actions()            
                    
                    net_actions = np.cumsum(actions,0)
                    
                else:
                    state = (env.state_seq[env.iter]-mu)/sigma
                    #state = env.state_seq[env.iter]
                    
                    ## get source actions:
                    actions = np.zeros((A.horizon,2))
                    
                    ## get net actions:
                    net_actions = np.zeros((A.horizon,2))
                                   
                    for i in range(1,A.horizon):
                                                
                        AS_n = np.concatenate((actions[i-1],state))
            
                        src_mu, log_sigma = sess.run([A.src_mu,A.src_log_sigma], feed_dict={ A.source_input_n: AS_n.reshape((1,4))})
                        
                        ## source action:
                        actions[i] = A.sampler(src_mu, log_sigma)
                        
                        ## net action:
                        net_actions[i] = actions[i] + net_actions[i-1]
                    
                env.iter += 1
                
                ## get responses from the environment:
                env.env_response(actions,A.horizon)
                                    
                ## group actions, initial state, and final state:                        
                axx_ = action_states(env,A,net_actions)
                
                mini_batch[horizon*i:horizon*(i+1)] = axx_
                action_batch[horizon*i:horizon*(i+1)] = actions
            
            ## normalise the state representations:
            mini_batch[:,2:6] = (mini_batch[:,2:6] - mu)/sigma
            
            train_feed_1 = {A.decoder_input_n : mini_batch,A.source_action : action_batch,\
                            A.prob : prob,A.lr:lr}
            
            sess.run(A.train_decoder,feed_dict = train_feed_1)
                
            # train source and critic:
            train_feed_2 = {A.beta: betas[count].reshape((1,1)), A.current_state: mini_batch[:,2:4],\
                            A.decoder_input_n : mini_batch, A.source_input_n : mini_batch[:,0:4], \
                            A.source_action : action_batch,
                            A.prob : prob,A.lr:lr}
            
            sess.run(A.train_critic_and_source,feed_dict = train_feed_2)
            
            #saver.save(sess,export_dir,global_step=count)
            
            if count % 500 == 0:   
                heatmap(0.1,sess,A,env,count,folder)
                
                
                        
        
if __name__ == "__main__":
    main()
    