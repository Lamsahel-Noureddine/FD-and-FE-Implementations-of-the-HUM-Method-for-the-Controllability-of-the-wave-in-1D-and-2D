# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:31:50 2021

@author: L
"""

import numpy as np

############## test of solver1Dellfini ##############

#The initial conditions 

def v(x):              # phi^0
    return x
    #return np.sin(np.pi*x)+x


def w(x):          # phi^1
    return x
    #return x


############## conjugate gradient  ##############

#The initial conditions 

def y0(x): 
    #return np.exp(-(5*(x-0.35))**6)
    #return np.sin(np.pi*x)
    #return np.exp(-x)-1
    return x
    '''
    XX=x.copy()
    i=0
    for ind in x:
        if ind<0.5:
            XX[i]=0
            i=i+1
        else:
            XX[i]=1
            i=i+1
    return XX            
     '''
 

    
def y1(x):
   #return np.sqrt(2)*np.cos(np.pi*x)
   return 0.*x
    
#initialization 

def ph0(x):              # phi^0
    return 0.*x        #0

def ph1(x):          # phi^1
    return 0.*x 
