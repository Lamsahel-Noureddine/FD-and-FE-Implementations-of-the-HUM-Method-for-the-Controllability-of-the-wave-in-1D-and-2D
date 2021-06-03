# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:21:28 2021

@author: noureddine lamsahel
"""
############## test of solve1d_wave ##############

#The initial conditions 

def v(x):              # phi^0
    return np.sin(np.pi*x)

def w(x):          # phi^1
    return 0.*x


############## conjugate gradient  ##############

#The initial conditions 

def y0(x): 
    #return np.exp(-(5*(x-0.37))**6)
    return np.sin(np.pi*x)
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
   return np.sqrt(2)*np.cos(np.pi*x)
    
#initialization 

def ph0(x):              # phi^0
    return 0.*x        #0

def ph1(x):          # phi^1
    return 0.*x 


