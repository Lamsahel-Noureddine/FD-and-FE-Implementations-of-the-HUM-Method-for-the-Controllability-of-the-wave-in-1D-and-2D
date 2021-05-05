# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:26:42 2021

@author: noureddine lamsahel
"""
import numpy as np

# Delta approximation 
def deltafunc(u,h):
    N=np.size(u)-1
    x=[]
    for i in range(0,N):
        termappi=(u[i+1]-u[i])/h
        x.append(termappi)
    
    termappN=(u[N]-u[N-1])/h
    x.append(termappN)
    deltau=np.array(x)
    return deltau
        
        
    



# integral approximation
def appInte(u,v,h):
    x=u*v
    intuv=h*np.sum(x)
    return intuv

def appintdelta(u,v,h):
    deltau=deltafunc(u,h)
    deltav=deltafunc(v,h)
    y=deltau*deltav
    intDeltauDeltav=h*np.sum(y)
    return intDeltauDeltav