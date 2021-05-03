# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:35:42 2021

@author: noureddine lamsahel
"""

import numpy as np
import scipy as sp
def solve1d(vj,wj,Un,r,k,M):
    N=np.size(vj)
    # The matrix of the approximate solution 
    A_final=np.zeros((N,M+2))
    
    # iteration 0
    phi0=vj
    
    # iteration 1
    phi1=phi0+k*wj
    
    A_final[:,0]=phi0
    A_final[:,1]=phi1
    
     # Construction from the explicit schema
    diag1=2*(1-r)*np.ones(N)
    diag2=r*np.ones(N-1)
    A=np.diag(diag1)+np.diag(diag2,1)+np.diag(diag2,-1)
    b0=(-1)*np.ones(N)
    b=np.diag(b0)
    c=np.zeros(N)
     # Descent 
    for n in range(1,M+1):
        c[-1]=r*Un[n]
        phi=sp.dot(A,phi1)+sp.dot(b,phi0)+c
        A_final[:,n+1]=phi
        phi0=phi1
        phi1=phi
        
    return A_final
        
