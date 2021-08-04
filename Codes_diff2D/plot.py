# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 07:54:55 2021

@author: L
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def plotControlfunc(y_final,x,y,M):
    xx=x[1:-1]
    yy=y[1:-1]
    tn=0
    X, Y = np.meshgrid(xx, yy)
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, y_final[tn], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #ax.set_zlim(1, 2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$ t=%d$'%(tn))
    
def plot_yT(y_final,x,y,M):
    xx=x[1:-1]
    yy=y[1:-1]
    tn=M+1
    X, Y = np.meshgrid(xx, yy)
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, y_final[tn], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #ax.set_zlim(1, 2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$ y(T)$')

def plot_y_prim_T(y_final,x,y,M,k):
    y_Mplus1=y_final[-1,:,:]
    y_M=y_final[M,:,:]
    y_pri_Mplus1=(1./k)*(y_Mplus1-y_M)
    xx=x[1:-1]
    yy=y[1:-1]
    X, Y = np.meshgrid(xx, yy)
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, y_pri_Mplus1, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #ax.set_zlim(1, 2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$y^{,}(T)$')
    

