# -*- coding: utf-8 -*-
"""
Created on Thu May 20 05:12:18 2021

@author: noureddine lamsahel
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D




def plotControlfunc(y_final,x,t):
    xx=x[:-1]
    Y, X = np.meshgrid(t, xx)

    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, X, y_final, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #ax.set_zlim(1, 2.5)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$y(x,t)$')
    
    
    
def plotposVit(y_final,x,M,h):
    xx=x[:-1]
    y_Mplus1=y_final[:,-1]
    y_M=y_final[:,M]
    y_pri_Mplus1=(y_Mplus1-y_M)/h
    '''
    plt.plot(xx,y_Mplus1)
    plt.xlabel('$x$')
    plt.ylabel('$y(T)$')
    '''
    plt.plot(xx,y_pri_Mplus1)
    plt.xlabel('$x$')
    plt.ylabel('$y^{\'}(T)$')
    

    