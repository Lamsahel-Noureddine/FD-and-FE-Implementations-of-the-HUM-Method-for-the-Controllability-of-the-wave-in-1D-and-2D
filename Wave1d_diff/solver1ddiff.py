# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:35:42 2021

@author: noureddine lamsahel
"""

import numpy as np
import scipy as sp
from Function_initial_Conditions import *
from Functions_util import *
from scipy.sparse import linalg , coo_matrix
#from scipy.sparse import *
from scipy import linalg as lg


 # solve the wave equation 
'''
 # Implicit scheme
def solve1d_wave(vj,wj,Un,h,k,M):
    N=np.size(vj)
    # The matrix of the approximate solution 
    A_final=np.zeros((N,M+2))
    
    # iteration 0
    phi0=vj
    
    # iteration 1
    phi1=phi0+k*wj
    
    A_final[:,0]=phi0
    A_final[:,1]=phi1
    
    a11=( 1./(k**2)+2./(h**2) )*np.ones(N)
    a12=( -1./(h**2) )*np.ones(N-1)
    A1=np.diag(a11)+np.diag(a12,1)+np.diag(a12,-1)
    c=np.zeros(N)
    for n in range(1,M+1):
        c[-1]=Un[n+1]
        b=(1./(k**2))*(2*phi1-phi0)+(1./(h**2))*c
        A1=coo_matrix(A1)
        A1=A1.tocsc()
        A1=A1.astype(np.float64)
        b=b.astype(np.float64)
        lu=linalg.splu(A1)
        phi=lu.solve(b)
        A_final[:,n+1]=phi
        
        phi0=phi1
        phi1=phi
        
 
    insb=np.zeros(M+2) 
    AA_final=np.insert(A_final,0,insb,axis=0)
    return AA_final

'''
 # explicit scheme
def solve1d_wave(vj,wj,Un,r,k,M):
    N=np.size(vj)
    # The matrix of the approximate solution 
    A_final=np.zeros((N,M+2))
    
    # iteration 0
    phi_0=vj
    
    # iteration 1
    phi_1=phi_0+k*wj
    
    A_final[:,0]=phi_0
    A_final[:,1]=phi_1
    
    for n in range(1,M+1):
        to1=2*(1-r)*A_final[0,n] + r*A_final[1,n] - A_final[0,n-1]
        A_final[0,n+1]=to1
        to2=2*(1-r)*A_final[N-1,n] + r*A_final[N-2,n] - A_final[N-1,n-1] + r*Un[n]
        A_final[N-1,n+1]=to2
        for j in range(1,N-1):
            ton=2*(1-r)*A_final[j,n] + r*A_final[j+1,n] + r*A_final[j-1,n] - A_final[j,n-1]
            A_final[j,n+1]=ton
  
    insb=np.zeros(M+2) 
    AA_final=np.insert(A_final,0,insb,axis=0)
    return AA_final



'''
 # theta_scheme
def solve1d_wave(vj,wj,Un,h,k,M):
    theta=0.5
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
    a11=( (1./(k**2))+(2*theta)/(h**2) )*np.ones(N)
    a12=( -theta/(h**2) )*np.ones(N-1)
    A1=np.diag(a11)+np.diag(a12,1)+np.diag(a12,-1)

    a21=( 2./(k**2) - (2*(1-2*theta)/(h**2)) )*np.ones(N)
    a22=((1-2*theta)/(h**2))*np.ones(N-1)
    A2=np.diag(a21)+np.diag(a22,1)+np.diag(a22,-1)
    a31=(2*theta/(h**2) - 1./(k**2))*np.ones(N)
    a32=(theta/(h**2) )*np.ones(N-1)
    A3=np.diag(a31)+np.diag(a32,1)+np.diag(a32,-1)
    A4=np.zeros(N)
    for n in range(1,M+1):
        phi=np.zeros(N)
        A4[-1]=(theta/(h**2))*Un[n+1]+((1-2*theta)/(h**2))*Un[n]+(theta/(h**2))*Un[n-1]
        bb=sp.dot(A2,phi1)+sp.dot(A3,phi0) + A4
        
        A1=coo_matrix(A1)
        A1=A1.tocsc()
        A1=A1.astype(np.float64)
        bb=bb.astype(np.float64)
        lu=linalg.splu(A1)
        phi=lu.solve(bb)
        
        A_final[:,n+1]=phi
        
        phi0=phi1
        phi1=phi
        
    insb=np.zeros(M+2) 
    AA_final=np.insert(A_final,0,insb,axis=0)
    return AA_final

'''

 #solve the laplacian equation 
def solve1d_Laplacian(psi0,psi1,h,k,vecy1):
    N=np.size(psi0)
    diag1=(-2.)*np.ones(N)
    diag2=np.ones(N-1)
    A=np.diag(diag1)+np.diag(diag2,1)+np.diag(diag2,-1)
    b=(-1)*(h**2/k)*(psi1-psi0)+(h**2)*vecy1
    
    A=coo_matrix(A)
    A=A.tocsc()
    A=A.astype(np.float64)
    b=b.astype(np.float64)
    #C=dsolve.spsolve(A,b,use_umfpack=False)
    lu=linalg.splu(A)
    C=lu.solve(b)
    CC=np.insert(C,0,0)
    return CC
    
    
def CG_alg(x,r,k,h,M):
    N=np.size(x)-2
    iterMAx=100000
    eps=1e-10
    #initialization
    vecphi0=ph0(x[:-1])
    vecphi1=ph1(x[:-1])

  # # # #  #  iteration 0  # # # # # # #
    #solve ph_0 system 
    matphi=solve1d_wave(vecphi0[1:],vecphi1[1:],np.zeros(M+2),r,k,M)  

    #solve psi_0 system 
    normalphi_1=(-1./h)*matphi[-1,:] 
    matpsi_int=solve1d_wave(np.zeros(N),np.zeros(N),normalphi_1[::-1],r,k,M)
           
    matpsi=matpsi_int[:,::-1]    
     #solve laplacian system
    vecy1=y1(x[1:-1])
    vecphi0_tilde=solve1d_Laplacian(matpsi[:,0][1:],matpsi[:,1][1:],h,k,vecy1)
    vecphi1_tilde=y0(x[:-1])-matpsi[:,0]
 
    cond_t2=np.sqrt(appintdelta(vecphi0_tilde,vecphi0_tilde,h)+appInte(vecphi1_tilde,vecphi1_tilde,h))
     #stopping criteria
    t1=np.sqrt(appintdelta(vecphi0_tilde,vecphi0_tilde,h)+appInte(vecphi1_tilde,vecphi1_tilde,h))
    t2=np.sqrt(appintdelta(vecphi0,vecphi0,h)+appInte(vecphi1,vecphi1,h))
    if t2==0.:
        t2=1.
    testcondition=t1/t2
    
    vecphi0_check=vecphi0_tilde.copy()
    vecphi1_check=vecphi1_tilde.copy()
    
    #  Descent
    itern=1
    while itern<=iterMAx and  testcondition > eps:
        print(itern)
        
        #solve ph_0_check system 
        matphi_check=solve1d_wave(vecphi0_check[1:],vecphi1_check[1:],np.zeros(M+2),r,k,M)
        normalphi_1_check=(-1./h)*matphi_check[-1,:]

        matpsi_check_int=solve1d_wave(np.zeros(N),np.zeros(N),normalphi_1_check[::-1],r,k,M)
        
        matpsi_check=matpsi_check_int[:,::-1] 
        
        #solve laplacian system
  
        vecphi0_line=solve1d_Laplacian(matpsi_check[:,0][1:],matpsi_check[:,1][1:],h,k,np.zeros(N))
        vecphi1_line=-matpsi_check[:,0]
        # calcul of rho_n
        term1=appintdelta(vecphi0_tilde,vecphi0_tilde,h)+appInte(vecphi1_tilde,vecphi1_tilde,h)
        term2=appintdelta(vecphi0_line,vecphi0_check,h)+appInte(vecphi1_line,vecphi1_check,h)
        rho=term1/term2
        #  #  #  #  go to n+1  #  #  #  # 
        vecphi0=vecphi0-rho*vecphi0_check
        vecphi1=vecphi1-rho*vecphi1_check
        matphi=matphi-rho*matphi_check
        matpsi=matpsi-rho*matpsi_check
             # for the calcul of gamma 
        gamma_vecphi0_tilde=vecphi0_tilde.copy()         #
        gamma_vecphi1_tilde=vecphi1_tilde.copy()
    
        vecphi0_tilde=vecphi0_tilde-rho*vecphi0_line
        vecphi1_tilde=vecphi1_tilde-rho*vecphi1_line
        
        #stopping criteria
        t1=np.sqrt(appintdelta(vecphi0_tilde,vecphi0_tilde,h)+appInte(vecphi1_tilde,vecphi1_tilde,h))
        print(testcondition)
        testcondition=t1/cond_t2
        
        
        
        
        # new descent direction
        ter1=appintdelta(vecphi0_tilde,vecphi0_tilde,h)+appInte(vecphi1_tilde,vecphi1_tilde,h)
        ter2=appintdelta(gamma_vecphi0_tilde,gamma_vecphi0_tilde,h)+appInte(gamma_vecphi1_tilde,gamma_vecphi1_tilde,h)
        gamma=ter1/ter2
    

        vecphi0_check=vecphi0_tilde+gamma*vecphi0_check
        vecphi1_check=vecphi1_tilde+gamma*vecphi1_check
        
         #  go to n+1
        
        itern=itern+1
        
    
        
    return vecphi0,vecphi1
            

def control_function(vecphi0_hat,vecphi1_hat,x,r,k,h,M):
    # Solve the ph_Hat system 
    matphi_hat=solve1d_wave(vecphi0_hat[1:],vecphi1_hat[1:],np.zeros(M+2),r,k,M)
    normalphi_hat_1=(-1./h)*matphi_hat[-1,:]
    y0j=y0(x[1:-1])
    y1j=y1(x[1:-1])
    # solve the control system
    maty_system_contr=solve1d_wave(y0j,y1j,normalphi_hat_1,r,k,M)
    return maty_system_contr
    
    
    
def get_norm(y_final,M,N,h,k):
    y_Mplus1=y_final[:,-1]
    y_M=y_final[:,M]
    y_pri_Mplus1=(y_Mplus1-y_M)/h
    
    norm_y_Mplus1=np.sqrt(appInte(y_Mplus1,y_Mplus1,h))
    print(norm_y_Mplus1)
    # norm of y^' at T in H^-1
    uu=solve1d_Laplacian(y_M[1:],y_Mplus1[1:],h,k,np.zeros(N))
    norm_y_Mplus1_prim=np.sqrt(appintdelta(uu,uu,h))
    print(norm_y_Mplus1_prim)    


def get_normcontrol(vecphi0_hat,vecphi1_hat,x,r,k,h,M):
    matphi_hat=solve1d_wave(vecphi0_hat[1:],vecphi1_hat[1:],np.zeros(M+2),r,k,M)
    normalphi_hat_1=(-1./h)*matphi_hat[-1,:]
    normvh=np.sqrt(appInte(normalphi_hat_1[:-1],normalphi_hat_1[:-1],k))
    print(normvh)
    return normalphi_hat_1
        
        
