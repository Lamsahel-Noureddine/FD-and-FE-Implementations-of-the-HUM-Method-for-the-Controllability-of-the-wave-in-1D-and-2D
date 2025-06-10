# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:40:05 2021

@author: Lamsahel
"""
import numpy as np

############## test of solver1Dellfini ##############

#exact solution of the wave equation 
def phi_exact(x,t):
    
    return np.cos(2*np.pi*t)*np.sin(2*np.pi*x)



def u(t):
    return 0*t

############## test of solve1d_Laplacian ##############

#exact solution of the laplacian  equation
def phi_exact_lapl(x):
    return (-1./(np.pi**2))*np.cos(np.pi*x)-(2./(np.pi**2))*x +(1/(np.pi**2))
