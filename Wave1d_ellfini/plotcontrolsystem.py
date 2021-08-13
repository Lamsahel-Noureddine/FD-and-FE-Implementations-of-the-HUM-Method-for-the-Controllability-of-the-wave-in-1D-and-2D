# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 13:54:41 2021

@author: L
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D




def plotControlfunc(y_final,x,t):
    Y, X = np.meshgrid(t, x)

    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, X, y_final, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #ax.set_zlim(1, 2.5)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$y(x,t)$')
    
    
    
def plotposVit(y_final,x,M,k):
    y_Mplus1=y_final[:,-1]
    y_M=y_final[:,M]
    y_pri_Mplus1=(y_Mplus1-y_M)/k
    '''
    plt.plot(xx,y_Mplus1)
    plt.xlabel('$x$')
    plt.ylabel('$y(T)$')
    '''
    plt.plot(x,y_pri_Mplus1)
    plt.xlabel('$x$')
    plt.ylabel('$y^{\'}(T)$')

def plotvcontrol(vcontrol,t):
    plt.plot(t,vcontrol)
    plt.xlabel('$t$')
    plt.ylabel('$v$')
    