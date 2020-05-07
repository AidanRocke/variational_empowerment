#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 18:56:50 2018

@author: aidanrocke & ildefonsmagrans
"""
import numpy as np
import tensorflow as tf

def get_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()
    
def dual_opt(var_name_1, var_name_2, loss, optimizer):
    
    vars_1 = tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope= var_name_1)
    train_1 = optimizer.minimize(loss,var_list=vars_1)
        
    vars_2 = tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope = var_name_2)
    train_2 = optimizer.minimize(loss,var_list=vars_2)
    
    return tf.group(train_1, train_2)

def action_states(env,agent,actions):

    ss_ = np.concatenate((env.state_seq[env.iter-agent.horizon-1],env.state_seq[env.iter-1])).reshape((1,4))
    S = np.repeat(ss_,agent.horizon,axis=0)
            
    return np.concatenate((actions,S),axis=1)

def gradient_norm(sess,name,loss,feed):
    
    var = tf.trainable_variables(name)
    
    gradients = tf.gradients(loss,[var[0]])[0]
    
    norm = tf.norm(gradients)
        
    return sess.run(norm, feed_dict=feed)
    