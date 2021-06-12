# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:11:57 2021

@author: EL BERDAI ADAM
"""
import numpy as np
from scipy import linalg as lg
from scipy.sparse import linalg , coo_matrix
#from scipy.sparse import *



def solver1Dellfini_band0(vj,wj,h,k,M):
    N=np.size(vj)
    # The matrix of the approximate solution 
    A_final=np.zeros((N,M+2))
    
    # iteration 0
    phi0=vj
    
    # iteration 1
    phi1=phi0+k*wj
    
    A_final[:,0]=phi0
    A_final[:,1]=phi1
    
    
    a1=(2./3)*np.ones(N)
    a2=(1./6)*np.ones(N-1)
    A=np.diag(a1)+np.diag(a2,-1)+np.diag(a2,1)
    
    b1=(2.)*np.ones(N)
    b2=(-1.)*np.ones(N-1)
    B=np.diag(b1)+np.diag(b2,-1)+np.diag(b2,1)
    
    BB=2*A-((k/h)**2)*B
    C=np.dot(lg.inv(A),BB)
    for n in range(2,M+2):
        phi=np.dot(C,phi1)-phi0
        A_final[:,n]=phi
        
        phi0=phi1
        phi1=phi
        
 
    insb=np.zeros(M+2) 
    AA_final=np.insert(A_final,0,insb,axis=0)
    return AA_final
    
    
def solver1Dellfini(vj,wj,h,k,M,Un):
    N=np.size(vj)
    # The matrix of the approximate solution 
    A_final=np.zeros((N,M+2))
    
    # iteration 0
    phi0=vj
    
    # iteration 1
    phi1=phi0+k*wj
    
    A_final[:,0]=phi0
    A_final[:,1]=phi1
    
    
    a1=(2./3)*np.ones(N)
    a2=(1./6)*np.ones(N-1)
    A=np.diag(a1)+np.diag(a2,-1)+np.diag(a2,1)
    
    b1=(2.)*np.ones(N)
    b2=(-1.)*np.ones(N-1)
    B=np.diag(b1)+np.diag(b2,-1)+np.diag(b2,1)
    
    BB=2*A-((k/h)**2)*B
    invA=lg.inv(A)
    C=np.dot(invA,BB)
    
    b=np.zeros(N)
    for n in range(1,M+1):
        bn_N=(-h/6*(k**2))*( Un[n+1]-2*Un[n]+Un[n-1] ) +(1/h)*Un[n]
        b[-1]=bn_N
        phi=np.dot(C,phi1)- phi0 +((k**2/h))*np.dot(invA,b)
        A_final[:,n+1]=phi
        phi0=phi1
        phi1=phi
        
 
    insb=np.zeros(M+2) 
    AA_final=np.insert(A_final,0,insb,axis=0)
    return AA_final
    


def solve1d_Laplacian(psi0,psi1,h,k,vecy1,normalphi_t1,normalphi_t0):
    N=np.size(psi0)
    a1=(2./3)*np.ones(N)
    a2=(1./6)*np.ones(N-1)
    A=np.diag(a1)+np.diag(a2,-1)+np.diag(a2,1)
    
    b1=(2.)*np.ones(N)
    b2=(-1.)*np.ones(N-1)
    B=np.diag(b1)+np.diag(b2,-1)+np.diag(b2,1)
    
    bb_1=np.zeros(N)
    bb_1[-1]=((h**2)/(6*k))*(normalphi_t1 - normalphi_t0)
    
    bb_2=(h**2)*vecy1
    psi01=psi1-psi0
    term_2= ((h**2)/k)*np.dot(A,psi01) + bb_1 - bb_2
    B=coo_matrix(B)
    B=B.tocsc()
    B=B.astype(np.float64)
    term_2=term_2.astype(np.float64)
    #C=dsolve.spsolve(A,b,use_umfpack=False)
    lu=linalg.splu(B)
    C_solution=lu.solve(term_2)
    CC_solution=np.insert(C_solution,0,0)
    return CC_solution
    
    
    
    
    
    
    
    
    
    
    

    


    
    
