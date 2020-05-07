#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:06:49 2018

@author: aidanrocke & ildefonsmagrans
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import imageio
import re

def heatmap(res,sess,agent,env,num_image,folder):
    """
    A function which plots a heatmap of the agent's empowerment critic at
    a particular resolution. 
    
    inputs:
        res: a floating point value between 0 and 1
        agent: a trained agent
    
    """
    plt.clf()
    # make sure that res is in the interval (0,1]:
    if res <= 0 or res > 1:
        raise Warning("resolution must satisfy 0 < res <= 1")
    
    ## create mesh grid:
    R, D = env.R, env.dimension
    xy = np.mgrid[R:int(D)-R:res, R:int(D)-R:res].reshape(2,-1).T
    L = int((D-2*R)/res)
    
    ## normalise the state representations:
    mu = env.dimension/2.0 - R ## mean of U(R,dimension-R)
    sigma = ((2*mu)**2)/12 ## variance of U(R,dimension-R)
    xy_ = (xy - mu)/sigma
    #xy_ = xy
    
    values = sess.run(agent.emp, feed_dict={agent.current_state: xy_})    
    sns.heatmap(values.reshape(L,L),xticklabels=False,\
                yticklabels=False,cmap="YlGnBu")
    
    plt.savefig(folder+str(num_image)+".png")
    
def variance_progression(X,variances,folder):

    plt.figure(figsize=(10,10))
    plt.style.use('ggplot')
    
    ## reshape the variances array:
    var = np.mean(variances,2)
        
    N = np.shape(variances)[0]
    
    for i in range(1,N):
        plt.plot(X,var[i], color=plt.cm.Blues(i/N), lw=3)
    
    plt.show()
    plt.savefig(folder+"variances"+".png")
    
def action_progression(X,actions,folder):

    plt.figure(figsize=(10,10))
    plt.style.use('ggplot')
    
    #act = np.mean(np.mean(actions,0),1)
    act = np.mean(actions,2)
    
    N = np.shape(actions)[0]
    
    for i in range(1,N):
        plt.plot(X,act[i], color=plt.cm.Blues(i/N), lw=3)
    
    plt.show()
    plt.savefig(folder+"actions"+".png")

    
def combined_loss(squared_loss,decoder_loss):
    
    plt.title('evolution of agent loss functions')
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

    f.set_size_inches(20,10)
    
    plot_1, = ax1.plot(squared_loss,label='squared_loss')
    
    ax1.legend(handles=[plot_1])
    
    plot_2, = ax2.plot(decoder_loss,label='decoder_loss')
    
    ax2.legend(handles=[plot_2])
    
    plt.show()
    
def plot_trajectory(env):
    """
    A function which gives us the last N locations of our agent superimposed
    as a scatter plot on top of the agent's entire trajectory. 
    """
    plt.clf()
    
    plt.grid(False)
    
    plt.plot(env.state_seq[:,0],env.state_seq[:,1],color='steelblue',zorder=0)
        
    start, finish = env.duration - 100, env.duration
    plt.scatter(env.state_seq[:,0][start:finish-1],env.state_seq[:,1][start:finish-1],color='crimson',linewidths=2,zorder=1)
    #plt.scatter(env.state_seq[:,0][start:finish-1],env.state_seq[:,1][start:finish-1],c=np.arange(100),cmap = 'Reds',linewidths=2,zorder=1)
    
    plt.show()
    
def spatial_histogram(env):
    """
    A spatial histogram of the last 100 locations of our agent. 
    """
    
    plt.title('spatial histogram and last 100 locations of agent')
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    f.set_size_inches(20,10)
    
    ax1.grid(False)
    ax2.grid(False)
    
    ax1.plot(env.state_seq[:,0],env.state_seq[:,1],color='steelblue',zorder=0)
    
    start, finish = env.duration - 100, env.duration
    
    ax1.scatter(env.state_seq[:,0][start:finish-1],env.state_seq[:,1][start:finish-1],color='crimson',linewidths=2,zorder=1)
    
    ax2.hist2d(env.state_seq[:,0], env.state_seq[:,1], bins=10,cmap='Blues')
    
    f.colorbar()
    
    ax2.colorbar()
    
    plt.show()
    
def sort_files(l):

  convert = lambda text: float(text) if text.isdigit() else text
  alphanum = lambda key: [ convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key) ]
  l.sort( key=alphanum )
  return l

    
def create_gif(folder,gif_destination):
    
    files = []
    
    z = sort_files(os.listdir(folder))

    for file in z[1:-1]:
        if file.endswith(".png"):
            files.append(os.path.join(folder, file))    

    images = []
    
    for filename in files:
        images.append(imageio.imread(filename))
        
    imageio.mimsave(gif_destination+'GIF.gif', images,duration = 2)