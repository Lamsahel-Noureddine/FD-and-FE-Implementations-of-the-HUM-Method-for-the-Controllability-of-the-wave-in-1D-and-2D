# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:40:05 2021

@author: L
"""
import numpy as np

############## test of solver1Dellfini ##############

#exact solution of the wave equation 
def phi_exact(x,t,N,M):
    phi_exactsol=np.zeros((N+2,M+2))
    for n in range(M+2):
        for i in range(N+2):
            phi_exactsol[i,n]=t[n]*x[i]+x[i]
            
    return phi_exactsol
    #return np.sin(np.pi*x)*np.cos(np.pi*t)+(t+1)*x




def u(t):
    return t+1
    #return t+1

############## test of solve1d_Laplacian ##############

#exact solution of the laplacian  equation
def phi_exact_lapl(x):
    res=(1./3)*(x**3)-(1/3)*x
    return res
    
    
    
