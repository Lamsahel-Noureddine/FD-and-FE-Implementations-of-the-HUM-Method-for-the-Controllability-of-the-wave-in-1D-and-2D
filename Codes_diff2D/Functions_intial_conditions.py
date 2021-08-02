# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 03:53:02 2021

@author: L
"""

import numpy as np
from numba import njit





############## test of solver_wave ##############

#The initial conditions 
@njit
def v(X,Y,N1,N2):            # phi^0
    sizX=N1
    sizY=N2
    Vij=np.zeros((sizX,sizY))
    for i in range(sizX):
        for j in range(sizY):
            Vij[i,j]=X[i]+1
    return Vij
@njit
def w(X,Y,N1,N2):          # phi^1
    sizX=N1
    sizY=N2
    Wij=np.zeros((sizX,sizY))
    for i in range(sizX):
        for j in range(sizY):
            Wij[i,j]=X[i]*Y[j]
    return Wij

############## conjugate gradient  ##############

#The initial conditions 

def y0(X,Y): 
    n1=np.size(X)
    n2=np.size(Y)
    Xx=np.zeros((n1,n2))
    rhoo=0.1

    for i in range(n1):
        for j in range(n2):
            
            r=(X[i]-0.35)**2+(Y[j]-0.35)**2
            Xx[i,j]=np.exp(-5*(r/(2*(rhoo**2)))**6)
            '''
            Xx[i,j]=np.sin(np.pi*X[i])*np.cos(np.pi*Y[j])
            '''
    return Xx

    
def y1(x,y):
    n1=np.size(x)
    n2=np.size(y)
    X=np.zeros((n1,n2))
    return X
    
#initialization 

def ph0(x,y):              # phi^0
    n1=np.size(x)
    n2=np.size(y)
    X=np.zeros((n1,n2))
    return X

def ph1(x,y):          # phi^1
    n1=np.size(x)
    n2=np.size(y)
    X=np.zeros((n1,n2))
    return X
