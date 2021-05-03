# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:38:39 2021

@author: noureddine lamsahel
"""
import numpy as np

#The boundary condition at 1
def u(x):
    return 0*x

#exact solution 
def phi_exact(x,t):
    term1=   np.sin(np.pi*(x+t))+np.sin(np.pi*(x-t))
    term2= (-1)*(np.sqrt(2)/np.pi)*np.cos(np.pi*(x+t)) +(np.sqrt(2)/np.pi)*np.cos(np.pi*(x-t))  
    return (1/2)*(term1+term2)


