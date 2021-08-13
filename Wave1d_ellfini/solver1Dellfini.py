# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:11:57 2021

@author: L
"""
import numpy as np
from scipy import linalg as lg
from scipy.sparse import linalg , coo_matrix
from Function_initial_Conditions import *
from Functions_util import *
from numba import njit
  

def solverlinear(matrA,vecb):
    matrixA=matrA.copy()
    vectb=vecb.copy()
    matrixA=coo_matrix(matrixA)
    matrixA=matrixA.tocsc()
    matrixA=matrixA.astype(np.float64)
    vectb=vectb.astype(np.float64)
    #C=dsolve.spsolve(A,b,use_umfpack=False)
    lu=linalg.splu(matrixA)
    C_sol=lu.solve(vectb)
    return C_sol






@njit
def prod_matrix_vectA0B0(A0B0,sec_term_temp,N):
    reslut=np.zeros(N)
    reslut[0]=A0B0[0,0]*sec_term_temp[0]+A0B0[0,1]*sec_term_temp[1]
    for i in range(1,N-1):
        reslut[i]=A0B0[0,1]*sec_term_temp[i-1]+A0B0[0,0]*sec_term_temp[i]+A0B0[0,1]*sec_term_temp[i+1]
    reslut[N-1]=A0B0[0,1]*sec_term_temp[N-2]+A0B0[0,0]*sec_term_temp[N-1]
    return reslut

    
def solver1Dellfini(vj,wj,C,k,M,Un,A0,A0B0):
    N=np.size(vj)
    # The matrix of the approximate solution 
    A_final=np.zeros((N,M+2))
    
    # iteration 0
    phi0=vj
    
    # iteration 1
    phi1=phi0+k*wj
    
    A_final[:,0]=phi0
    A_final[:,1]=phi1
    Cn=np.zeros(N)
    for n in range(1,M+1):
        Cn_N=(-1.)*( Un[n+1]+Un[n-1] ) +(( 6*(C**2))+2)*Un[n]
        Cn[-1]=Cn_N
        term_scd=prod_matrix_vectA0B0(A0B0, phi1,N)- prod_matrix_vectA0B0(A0, phi0,N)+Cn
        phi=solverlinear(A0,term_scd)
        A_final[:,n+1]=phi
        phi0=phi1
        phi1=phi
        
    finalA=np.vstack((A_final,Un))
    fn_A=np.vstack((np.zeros(M+2),finalA))
    return fn_A
    


def solve1d_Laplacian(psi0,psi1,h,k,vecy1,A0,B0,N):
    b_L=np.zeros(N)
    b_L[0]=-((h**2)/6.)*vecy1[0]
    b_L[-1]=((h**2)/6.)*( (  (psi1[-1] - psi0[-1])/k  )-vecy1[-1] )
    
    Y= (  (1./k)*(psi1[1:-1] - psi0[1:-1])  )-vecy1[1:-1] 
    
    term_L= ((h**2)/6.)*prod_matrix_vectA0B0(A0, Y,N) + b_L
    
    C_solution=solverlinear(B0,term_L)
    
    C_solution=np.append(C_solution,0)
    C_solution=np.append(0,C_solution)
    return C_solution
    
    
    
def CG_alg(x,k,h,C,M,A0,B0,A0B0):
    N=np.size(x)-2
    iterMAx=2000
    eps=1e-10
    #initialization
    vecphi0=ph0(x)
    vecphi1=ph1(x)

  # # # #  #  iteration 0  # # # # # # #
    #solve ph_0 system 
    matphi=solver1Dellfini(vecphi0[1:-1],vecphi1[1:-1],C,k,M,np.zeros(M+2),A0,A0B0)  
    #solve psi_0 system 
    normalphi_1=(-1./h)*matphi[N,:] 
    matpsi_int=solver1Dellfini(np.zeros(N),np.zeros(N),C,k,M,normalphi_1[::-1],A0,A0B0)   
    matpsi=matpsi_int[:,::-1]    
     #solve laplacian system
    vecy1=y1(x)
    vecphi0_tilde=solve1d_Laplacian(matpsi[:,0],matpsi[:,1],h,k,vecy1,A0,B0,N)
    vecphi1_tilde=y0(x)-matpsi[:,0]
    
    cond_t2=np.sqrt(appintdelta(vecphi0_tilde,vecphi0_tilde,h,N)+appInte(vecphi1_tilde,vecphi1_tilde,h,N))
     #stopping criteria
    t1=np.sqrt(appintdelta(vecphi0_tilde,vecphi0_tilde,h,N)+appInte(vecphi1_tilde,vecphi1_tilde,h,N))
    t2=np.sqrt(appintdelta(vecphi0,vecphi0,h,N)+appInte(vecphi1,vecphi1,h,N))
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
        matphi_check=solver1Dellfini(vecphi0_check[1:-1],vecphi1_check[1:-1],C,k,M,np.zeros(M+2),A0,A0B0)
        normalphi_1_check=(-1./h)*matphi_check[N,:]

        matpsi_check_int=solver1Dellfini(np.zeros(N),np.zeros(N),C,k,M,normalphi_1_check[::-1],A0,A0B0)
        
        matpsi_check=matpsi_check_int[:,::-1] 
        
        #solve laplacian system
  
        vecphi0_line=solve1d_Laplacian(matpsi_check[:,0],matpsi_check[:,1],h,k,np.zeros(N+2),A0,B0,N)
        vecphi1_line=-matpsi_check[:,0]
        # calcul of rho_n
        term1=appintdelta(vecphi0_tilde,vecphi0_tilde,h,N)+appInte(vecphi1_tilde,vecphi1_tilde,h,N)
        term2=appintdelta(vecphi0_line,vecphi0_check,h,N)+appInte(vecphi1_line,vecphi1_check,h,N)
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
        t1=np.sqrt(appintdelta(vecphi0_tilde,vecphi0_tilde,h,N)+appInte(vecphi1_tilde,vecphi1_tilde,h,N))
        print(testcondition)
        testcondition=t1/cond_t2
        
        
        
        
        # new descent direction
        ter1=appintdelta(vecphi0_tilde,vecphi0_tilde,h,N)+appInte(vecphi1_tilde,vecphi1_tilde,h,N)
        ter2=appintdelta(gamma_vecphi0_tilde,gamma_vecphi0_tilde,h,N)+appInte(gamma_vecphi1_tilde,gamma_vecphi1_tilde,h,N)
        gamma=ter1/ter2
    

        vecphi0_check=vecphi0_tilde+gamma*vecphi0_check
        vecphi1_check=vecphi1_tilde+gamma*vecphi1_check
        
         #  go to n+1
        
        itern=itern+1
        
    
        
    return vecphi0,vecphi1

            
def control_function(vecphi0_hat,vecphi1_hat,x,k,h,C,M,N,A0,A0B0):
    # Solve the ph_Hat system 
    matphi_hat=solver1Dellfini(vecphi0_hat[1:-1],vecphi1_hat[1:-1],C,k,M,np.zeros(M+2),A0,A0B0) 
    normalphi_hat_1=(-1./h)*matphi_hat[N,:]
    y0j=y0(x)
    y1j=y1(x)
    # solve the control system
    maty_system_contr=solver1Dellfini(y0j[1:-1],y1j[1:-1],C,k,M,normalphi_hat_1,A0,A0B0)
    return maty_system_contr
    
    
    
def get_norm(y_final,M,N,h,k,A0,B0):
    y_Mplus1=y_final[:,-1]
    y_M=y_final[:,M]
    y_pri_Mplus1=(y_Mplus1-y_M)/k
    
    norm_y_Mplus1=np.sqrt(appInte(y_Mplus1,y_Mplus1,h,N))
    print(norm_y_Mplus1)
    # norm of y^' at T in H^-1
    uu=solve1d_Laplacian(y_M,y_Mplus1,h,k,np.zeros(N+2),A0,B0,N)
    norm_y_Mplus1_prim=np.sqrt(appintdelta(uu,uu,h,N))
    print(norm_y_Mplus1_prim)    



def get_normcontrol(vecphi0_hat,vecphi1_hat,x,C,k,h,M,N,A0,A0B0):
    matphi_hat=solver1Dellfini(vecphi0_hat[1:-1],vecphi1_hat[1:-1],C,k,M,np.zeros(M+2),A0,A0B0)
    normalphi_hat_1=(-1./h)*matphi_hat[N,:]
    normvh=np.sqrt(appInte(normalphi_hat_1,normalphi_hat_1,k,M))
    print(normvh)
    return normalphi_hat_1


    
    
           
    
    
    
    
    
    
    

    


    
    
