# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:26:42 2021

@author: noureddine lamsahel
"""
import numpy as np




# Delta approximation 
def deltafunc(U,h):
    N=np.size(U)-1
    xtemp=[]
    termapp0=termappi=(U[1]-U[0])/h
    xtemp.append(termapp0)
    for i in range(1,N):
        termappi=(U[i+1]-U[i-1])/(2*h)
        xtemp.append(termappi)
    
    termappN=(U[N]-U[N-1])/h
    xtemp.append(termappN)
    deltau=np.array(xtemp)
    return deltau
        
        
    



# integral approximation
def appInte(uu,vv,h):
    xtemp=uu*vv
    intuv=h*np.sum(xtemp)
    return intuv

def appintdelta(uu,vv,h):
    deltau=deltafunc(uu,h)
    deltav=deltafunc(vv,h)
    ytemp=deltau*deltav
    intDeltauDeltav=h*np.sum(ytemp)
    return intDeltauDeltav



    
    
    
    
    
    
    
    
    
    