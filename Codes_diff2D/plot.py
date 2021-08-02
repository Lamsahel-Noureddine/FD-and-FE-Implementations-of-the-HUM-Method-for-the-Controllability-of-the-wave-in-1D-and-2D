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