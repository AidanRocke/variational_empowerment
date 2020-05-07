#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 10:46:54 2017

@author: aidanrocke & ildefonsmagrans
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from utils import dual_opt

class agent_cognition:
    
    """
        An agent that reasons using a measure of empowerment. 
        Here we assume that env refers to an initialised environment class. 
    """
    
    def __init__(self,planning_horizon,seed, bound):
        self.seed = seed
        self.horizon = planning_horizon        
        self.bound = bound
        
        self.current_state = tf.placeholder(tf.float32, [None, 2])
        self.source_action = tf.placeholder(tf.float32, [None, 2])
        # define a placeholder for beta values in the squared loss:
        self.beta = tf.placeholder(tf.float32, [None, 1])
        
        ## define a placeholder for the dropout value:
        self.prob = tf.placeholder_with_default(1.0, shape=(),name='prob')
        
        ## define a placeholder for the learning rate:
        self.lr = tf.placeholder(tf.float32, shape = [],name='lr')
        
        ## define empowerment critic:
        self.emp = self.empowerment_critic()
                
        ## define source:
        self.source_input_n = tf.placeholder(tf.float32, [None, 4],name='src_input')
        
        self.src_mu, self.src_log_sigma = self.source_dist_n()
        self.src_dist = tfp.distributions.MultivariateNormalDiag(self.src_mu, \
                                                             tf.exp(self.src_log_sigma))
                            
        self.log_src = tf.identity(self.src_dist.log_prob(self.source_action),name='log_src')
        
        ## define decoder parameters and log probability:
        self.decoder_input_n = tf.placeholder(tf.float32, [None, 6])
        self.decoder_mu, self.decoder_log_sigma = self.decoder_dist_n()
        
        self.decoder_dist = tfp.distributions.MultivariateNormalDiag(self.decoder_mu, \
                                                             tf.exp(self.decoder_log_sigma))
        
        self.log_decoder = self.decoder_dist.log_prob(self.source_action)
        
        ## define losses:
        self.decoder_loss = tf.reduce_mean(-1.0*self.log_decoder)
        self.squared_loss = tf.reduce_mean(tf.square(self.beta*self.log_decoder-self.emp-self.log_src))
        
        ### define the optimisers:
        self.fast_optimizer = tf.train.AdagradOptimizer(self.lr,name='ada_1')
        self.slow_optimizer = tf.train.AdagradOptimizer(self.lr,name='ada_2')
        
        self.train_decoder = self.fast_optimizer.minimize(self.decoder_loss)
        
        ### define a dual optimizatio method for critic and source:
        self.train_critic_and_source = dual_opt("critic", "source", self.squared_loss,\
                                                self.slow_optimizer)
        
    
        self.init_g = tf.global_variables_initializer() 
    
    def init_weights(self,shape,var_name):
        """
            Xavier initialisation of neural networks
        """
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape),name = var_name)
        
    def two_layer_net(self, X, w_h, w_h2, w_o,bias_1, bias_2):
        """
            A generic method for creating two-layer networks
            
            input: weights
            output: neural network
        """
        
        h = tf.nn.elu(tf.add(tf.matmul(X, w_h),bias_1))
        drop_1 = tf.nn.dropout(h, self.prob)
        
        h2 = tf.nn.elu(tf.add(tf.matmul(drop_1, w_h2),bias_2))
        drop_2 = tf.nn.dropout(h2, self.prob)
        
        return tf.matmul(drop_2, w_o)
    
    def empowerment_critic(self):
        """
        This function provides a cheap approximation to empowerment
        upon convergence of the training algorithm. Given that the 
        mutual information is non-negative this function must only
        give non-negative output. 
        
        input: state
        output: empowerment estimate
        """
        
        #with tf.variable_scope("critic",reuse=tf.AUTO_REUSE):
        with tf.variable_scope("critic"):
            
            tf.set_random_seed(self.seed)
    
            w_h = self.init_weights([2,500],"w_h")
            w_h2 = self.init_weights([500,300],"w_h2")
            w_o = self.init_weights([300,1],"w_o")
            
            ### bias terms:
            bias_1 = self.init_weights([500],"bias_1")
            bias_2 = self.init_weights([300],"bias_2")
            bias_3 = self.init_weights([1],"bias_3")
                
            h = tf.nn.elu(tf.add(tf.matmul(self.current_state, w_h),bias_1))
            h2 = tf.nn.elu(tf.add(tf.matmul(h, w_h2),bias_2))
        
        return tf.nn.elu(tf.add(tf.matmul(h2, w_o),bias_3))
        
        
    def source_dist_n(self):
        
        """
            This is the per-action source distribution, also known as the 
            exploration distribution. 
        """
        
        #with tf.variable_scope("source",reuse=tf.AUTO_REUSE):
        with tf.variable_scope("source"):
                               
            tf.set_random_seed(self.seed)
            
            W_h = self.init_weights([4,300],"W_h")
            W_h2 = self.init_weights([300,100],"W_h2")
            W_o = self.init_weights([100,10],"W_o")
            
            # define bias terms:
            bias_1 = self.init_weights([300],"bias_1")
            bias_2 = self.init_weights([100],"bias_2")
            
            ## two-layer network:
            h = tf.nn.elu(tf.add(tf.matmul(self.source_input_n, W_h),bias_1))
            drop_1 = tf.nn.dropout(h, self.prob)
        
            h2 = tf.nn.elu(tf.add(tf.matmul(drop_1, W_h2),bias_2))
            drop_2 = tf.nn.dropout(h2, self.prob)
            
            Tau = tf.matmul(drop_2, W_o)
                                    
            W_mu = self.init_weights([10,2],"W_mu")
            W_sigma = self.init_weights([10,2],"W_sigma")
            
            mu = tf.multiply(tf.nn.tanh(tf.matmul(Tau,W_mu)),self.bound)
            log_sigma = tf.multiply(tf.nn.tanh(tf.matmul(Tau,W_sigma)),self.bound)
        
        return mu, log_sigma
    
    
    def sampler(self,mu,log_sigma):
                        
        return np.random.normal(mu,np.exp(log_sigma))   
    
    def random_actions(self):
        """
            This baseline is used as a drop in replacement for the source at the
            early stages of learning and to check that the source isn't completely useless. 
        """
        
        return np.random.normal(0,self.bound,size = (self.horizon,2))
        
    def decoder_dist_n(self): 
        
        """
            This is the per-action decoder, also known as the 
            planning distribution. 
        """
        
        #with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
        with tf.variable_scope("decoder"):
            
            tf.set_random_seed(self.seed)
                        
            W_h = self.init_weights([6,300],"W_h")
            W_h2 = self.init_weights([300,100],"W_h2")
            W_o = self.init_weights([100,10],"W_o")
            
            # define bias terms:
            bias_1 = self.init_weights([300],"bias_1")
            bias_2 = self.init_weights([100],"bias_2")
            
            ## two-layer network:
            h = tf.nn.elu(tf.add(tf.matmul(self.decoder_input_n, W_h),bias_1))
            h2 = tf.nn.elu(tf.add(tf.matmul(h, W_h2),bias_2))
            
            Tau = tf.matmul(h2, W_o)
                        
            W_mu = self.init_weights([10,2],"W_mu")
            W_sigma = self.init_weights([10,2],"W_sigma")
            
            mu = tf.multiply(tf.nn.tanh(tf.matmul(Tau,W_mu)),self.bound)
            log_sigma = tf.multiply(tf.nn.tanh(tf.matmul(Tau,W_sigma)),self.bound)
                    
            
        return mu, log_sigma