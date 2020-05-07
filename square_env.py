#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 10:47:40 2017

@author: aidanrocke & ildefonsmagrans
"""

import numpy as np

class square_env:
    def __init__(self,duration,radius,dimension):
        if 2*radius > dimension:
            raise Warning("diameter can't exceed dimensions")
        self.R = radius # radius of agent
        self.dimension = dimension # LxW of the square world
        self.eps = radius/100
        self.lower_limit, self.upper_limit = self.R+self.eps, self.dimension-self.R-self.eps
        self.iter = 0 # current iteration
        self.duration = duration # maximum duration of agent in environment
        self.state_seq = np.zeros((self.duration,2))
                
    def random_initialisation(self):
        # method for initialisation: 
            
        self.state_seq[self.iter][0] = np.random.uniform(self.lower_limit, self.upper_limit)
        self.state_seq[self.iter][1] = np.random.uniform(self.lower_limit, self.upper_limit)
        
        self.iter = 1
        
    def boundary_conditions(self):
                
        #boundary conditions:
        cond_X = (self.state_seq[self.iter-1][0] >= self.lower_limit)*(self.state_seq[self.iter-1][0] <= self.upper_limit)
        cond_Y = (self.state_seq[self.iter-1][1] >= self.lower_limit)*(self.state_seq[self.iter-1][1] <= self.upper_limit)

        return cond_X*cond_Y
            
    def step(self, action):
                
        self.state_seq[self.iter] = self.state_seq[self.iter-1] + action
        
        #return to previous state if boundary conditions are not satisfied:
        if self.boundary_conditions() == 0:
            self.state_seq[self.iter] -= action
            
        self.iter += 1
            
        if self.iter > self.duration:
            raise Exception("Game over!")            
            
            
    def env_response(self,actions,horizon):
        # update the environment
        
        for i in range(1,horizon):
            self.step(actions[i])
        
        
    def reset(self):
        """
        Return to the initial conditions. 
        """
        self.state_seq = np.zeros((self.duration,2))
        self.iter = 0
        
        
        
    

    

