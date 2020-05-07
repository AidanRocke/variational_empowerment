#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 00:49:46 2018

@author: aidanrocke & ildefonsmagrans
"""
import unittest
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')

from agent import agent_cognition
from square_env import square_env

env = square_env(1000,1)
A = agent_cognition(7)

## define functions to be tested:

critic = A.empowerment_critic()

decoder_dist = A.decoder_distribution()

source_dist = A.source_distribution()

sampler = A.sampler

source = A.source

decoder = A.decoder

class agent_tests(unittest.TestCase):

    def test_critic(self):
        
        
        X = np.random.rand(2)
        
        with tf.Session() as sess:
            
            init = tf.global_variables_initializer().run()
                                            
            out = sess.run(critic, feed_dict={A.X1: X.reshape((1,2))})
            
            self.assertTrue(np.shape(out) == (1,1))
            
    def test_decoder_dist(self):
        
        X = np.random.rand(6)
        
        with tf.Session() as sess:
            
            init = tf.global_variables_initializer().run()
            
            x,y = sess.run(decoder_dist, feed_dict={A.X2: X.reshape((1,6))})
            
            self.assertTrue((np.shape(x) == (1,2)) & (np.shape(y) == (1,2)))
            
    def test_source_dist(self):
        
        X = np.random.rand(4)
        
        with tf.Session() as sess:
                
            init = tf.global_variables_initializer().run()
            
            x,y = sess.run(source_dist, feed_dict={A.X3: X.reshape((1,4))})
            
            self.assertTrue((np.shape(x) == (1,2)) & (np.shape(y) == (1,2)))
            
    def test_sampler(self):
        
        X = np.random.rand(4)
    
        with tf.Session() as sess:
                
            init = tf.global_variables_initializer().run()
            
            x,y = sess.run(source_dist, feed_dict={A.X3: X.reshape((1,4))})
            
            z = sess.run(sampler(x,y))
            
            self.assertTrue(np.shape(z) == (1,2))
            
    def test_source(self):
        
        a_, s_ = np.random.rand(2), np.random.rand(2)
    
        with tf.Session() as sess:
            
            init = tf.global_variables_initializer().run()
                    
            y = source(a_,s_,sess)
            
            self.assertTrue(type(y[0]) == np.float32)
            
    def test_decoder(self):
        
        a_, s_,s = np.random.rand(2), np.random.rand(2), np.random.rand(2)
    
        with tf.Session() as sess:
            
            init = tf.global_variables_initializer().run()
                    
            y = decoder(a_, s_, s, sess)
            
            self.assertTrue(type(y[0]) == np.float32)
            
    

if __name__ == '__main__':
    unittest.main()


## random test:
    
a_, s_ = np.random.rand(2,3), np.random.rand(2,3)
    
with tf.Session() as sess:
    
    init = tf.global_variables_initializer().run()
                
    y = source(a_,s_,sess)