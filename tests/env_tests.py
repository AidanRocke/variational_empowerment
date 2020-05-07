#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:35:55 2018

@author: aidanrocke & ildefonsmagrans
"""

## testing the environment:
import numpy as np
from square_env import square_env

class Environment_Test:
    
    def __init__(self):
        self.square = square_env(0.01,100,(20.0,10.0))
        self.square.initial_conditions()
        
    def stepper(self,i,a,b):
        return self.square.state_seq[i]
            
    def step_test(self):
        z = 0
        for i in range(99):
            
            a,b = np.random.binomial(max(self.square.dims),0.5) , np.random.uniform(0,2*np.pi)
            
            self.square.step((a,b))
            
            cond_X = (self.square.state_seq[i][0] >= 0)*(self.square.state_seq[i][0] <= self.square.dims[0])
            cond_Y = (self.square.state_seq[i][1] >= 0)*(self.square.state_seq[i][1] <= self.square.dims[1])
            
            if cond_X*cond_Y == 1:
                z += self.stepper(i,a,b) == self.square.state_seq[i]
                
            else:
                z += self.stepper(i,a,b) == self.square.state_seq[i-1]
                
        return z/99
        
if __name__ == '__main__':
    env= Environment_Test()
    print(env.step_test()==1.00)
 