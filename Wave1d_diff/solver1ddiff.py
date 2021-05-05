# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:35:42 2021

@author: noureddine lamsahel
"""

import numpy as np
import scipy as sp

from scipy.sparse import linalg , coo_matrix
#from scipy.sparse import *



 # solve the wave equation 
def solve1d_wave(vj,wj,Un,r,k,M):
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

    c=np.zeros(N)
     # Descent 
    for n in range(1,M+1):
        c[-1]=r*Un[n]
        phi=sp.dot(A,phi1)-phi0 + c
        A_final[:,n+1]=phi
        phi0=phi1
        phi1=phi
    insb=np.zeros(M+2)   
    AA_final=np.insert(A_final,0,insb,axis=0)
    return AA_final

 #solve the laplacian equation 
def solve1d_Laplacian(psi0,psi1,h,k,y1):
    N=np.size(psi0)
    
    diag1=(-2)*np.ones(N)
    diag2=np.ones(N-1)
    A=np.diag(diag1)+np.diag(diag2,1)+np.diag(diag2,-1)
    b=(h**2/k)*(psi1-psi0)-(h**2)*y1
    
    A=coo_matrix(A)
    A=A.tocsc()
    A=A.astype(np.float64)
    b=b.astype(np.float64)
    #C=dsolve.spsolve(A,b,use_umfpack=False)
    lu=linalg.splu(A)
    C=lu.solve(b)
    CC=np.insert(C,0,0)
    return CC
        
