# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 22:52:13 2021

@author: L
"""
import numpy as np
from Solver2Dellfini import *
from numba import njit
from assembling_global import *
from initial_conditions import *
from  Functions_use import *

@njit
def get_normal(PH,N1,N2,h1,h2,M):
    N_I=N1*N2
    N_T=(N1+2)*(N2+2)
    N_B=2*(N1+N2)+4
    PH_dervnorma=np.zeros((N_B,M+2))
    for n in range(M+2):
        PH_n=PH[:,n]
        PH_nor=[0]
        for j in range(N2):
            PH_nor.append((-1/h1)*PH_n[j]  )

        PH_nor.append(0)
        
        for i in range(N1):
            PH_nor.append(  (-1/h2)*PH_n[i*N2]  )
            PH_nor.append(  (-1/h2)*PH_n[(i+1)*N2-1]  )

        PH_nor.append(0)
        for j in range(N2):
            PH_nor.append((-1/h1)*PH_n[(N_I-1)-(N2-1):][j]  )


        PH_nor.append(0)
        PH_dervnorma[:,n]=np.array(PH_nor)
      
    return PH_dervnorma
    
    

def CG_alg(nodes,tabl_inter_bond,tabl_bond,tabl_inter,AB,A_inter,B_inter,A_glabal,B_glabal,k,N1,N2,M):
    N_I=N1*N2
    N_T=(N1+2)*(N2+2)
    N_B=2*(N1+N2)+4
    h1=1/(N1+1)
    h2=1/(N2+1)
    

    iterMAx=2000
    eps=1e-2
    #initialization
    vecphi0=ph0(nodes,N_T,tabl_inter_bond)
    vecphi1=ph1(nodes,N_T,tabl_inter_bond)

  # # # #  #  iteration 0  # # # # # # #
    #solve ph_0 system 
    C_sm=np.zeros((N_I,M))
    matphi=solver_wave2D(AB,A_inter,k,vecphi0[:N_I],vecphi1[:N_I],C_sm,N_I,M,N2,N1,np.zeros((N_B,M+2))) 
    
    
    #solve psi_0 system 
    Un=get_normal(matphi,N1,N2,h1,h2,M)
    C_sm=second_member(A_glabal,B_glabal,tabl_inter,tabl_bond,Un[:,::-1],M,N_I,N_B,k)
    matpsi_int=solver_wave2D(AB,A_inter,k,np.zeros(N_I),np.zeros(N_I),C_sm,N_I,M,N2,N1,Un[:,::-1])   
    matpsi=matpsi_int[:,::-1]   
    
    
     #solve laplacian system
    Psi0=matpsi[:,0]
    Psi1=matpsi[:,1]
    Y_vec=y1(nodes,N_T,tabl_inter_bond)
    C_sm_l=second_member_lap(A_glabal,tabl_inter,tabl_bond,Psi0,Psi1,Y_vec,M,N_I,N_B,k)
    vecphi0_tilde=Solver_lapl(A_inter,B_inter,C_sm_l,N_I,N_B,N2,N1,k,Psi0,Psi1,Y_vec)
    vecphi1_tilde=y0(nodes,N_T,tabl_inter_bond)-Psi0
    
    cond_t2=np.sqrt(integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,vecphi0_tilde,vecphi0_tilde,h1,h2)+integral_phi_psi(tabl_inter_bond,A_glabal,N_T,vecphi1_tilde,vecphi1_tilde,h1,h2))
    
     #stopping criteria
    t1=np.sqrt(integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,vecphi0_tilde,vecphi0_tilde,h1,h2)+integral_phi_psi(tabl_inter_bond,A_glabal,N_T,vecphi1_tilde,vecphi1_tilde,h1,h2))
    t2=np.sqrt(integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,vecphi0,vecphi0,h1,h2)+integral_phi_psi(tabl_inter_bond,A_glabal,N_T,vecphi1,vecphi1,h1,h2))
    
    print(t1)
    if t2==0.:
        t2=1.
    testcondition=t1/t2
    
    vecphi0_check=vecphi0_tilde.copy()
    vecphi1_check=vecphi1_tilde.copy()
    
    #  Descent
    itern=1
    print('HEY bro')
    while itern<=iterMAx and  testcondition > eps:
        print(itern)
        
        #solve ph_0_check system 
        C_sm=np.zeros((N_I,M))
        matphi_check=solver_wave2D(AB,A_inter,k,vecphi0_check[:N_I],vecphi1_check[:N_I],C_sm,N_I,M,N2,N1,np.zeros((N_B,M+2))) 
        
        #solve psi_0_check system 
        Un=get_normal(matphi_check,N1,N2,h1,h2,M)
        C_sm=second_member(A_glabal,B_glabal,tabl_inter,tabl_bond,Un[:,::-1],M,N_I,N_B,k)
        matpsi_check_int=solver_wave2D(AB,A_inter,k,np.zeros(N_I),np.zeros(N_I),C_sm,N_I,M,N2,N1,Un[:,::-1])   
        matpsi_check=matpsi_check_int[:,::-1]  
        
        #solve laplacian system
        Psi0=matpsi_check[:,0]
        Psi1=matpsi_check[:,1]
        Y_vec=np.zeros(N_T)
        C_sm_l=second_member_lap(A_glabal,tabl_inter,tabl_bond,Psi0,Psi1,Y_vec,M,N_I,N_B,k)
        vecphi0_line=Solver_lapl(A_inter,B_inter,C_sm_l,N_I,N_B,N2,N1,k,Psi0,Psi1,Y_vec)
        vecphi1_line=-Psi0
        
        # calcul of rho_n
        term1=integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,vecphi0_tilde,vecphi0_tilde,h1,h2)+integral_phi_psi(tabl_inter_bond,A_glabal,N_T,vecphi1_tilde,vecphi1_tilde,h1,h2)
        term2=integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,vecphi0_line,vecphi0_check,h1,h2)+ integral_phi_psi(tabl_inter_bond,A_glabal,N_T,vecphi1_line,vecphi1_check,h1,h2)
        
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
        t1=np.sqrt(integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,vecphi0_tilde,vecphi0_tilde,h1,h2)+integral_phi_psi(tabl_inter_bond,A_glabal,N_T,vecphi1_tilde,vecphi1_tilde,h1,h2))
        print(testcondition)
        testcondition=t1/cond_t2
        
        
        
        
        # new descent direction
        ter1=integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,vecphi0_tilde,vecphi0_tilde,h1,h2)+integral_phi_psi(tabl_inter_bond,A_glabal,N_T,vecphi1_tilde,vecphi1_tilde,h1,h2)
        ter2=integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,gamma_vecphi0_tilde,gamma_vecphi0_tilde,h1,h2)+integral_phi_psi(tabl_inter_bond,A_glabal,N_T,gamma_vecphi1_tilde,gamma_vecphi1_tilde,h1,h2)
        gamma=ter1/ter2
    

        vecphi0_check=vecphi0_tilde+gamma*vecphi0_check
        vecphi1_check=vecphi1_tilde+gamma*vecphi1_check
        
         #  go to n+1
        
        itern=itern+1
        
    
        
    return vecphi0,vecphi1

def control_function(vecphi0_hat,vecphi1_hat,nodes,tabl_bond,tabl_inter_bond,tabl_inter,AB,A_inter,A_glabal,B_glabal,k,N1,N2,M):
    N_I=N1*N2
    N_B=2*(N1+N2)+4
    N_T=(N1+2)*(N2+2)
    h1=1./(N1+1)
    h2=1./(N2+1)
    # Solve the ph_Hat system 
    C_sm=np.zeros((N_I,M))
    matphi_hat=solver_wave2D(AB,A_inter,k,vecphi0_hat[:N_I],vecphi1_hat[:N_I],C_sm,N_I,M,N2,N1,np.zeros((N_B,M+2))) 
    
    y0j=y0(nodes,N_T,tabl_inter_bond)
    y1j=y1(nodes,N_T,tabl_inter_bond)
    
    # solve the control system
    Un=get_normal(matphi_hat,N1,N2,h1,h2,M)
    C_sm=second_member(A_glabal,B_glabal,tabl_inter,tabl_bond,Un,M,N_I,N_B,k)
    maty_system_contr=solver_wave2D(AB,A_inter,k,y0j[:N_I],y0j[:N_I],C_sm,N_I,M,N2,N1,Un) 
    
    return maty_system_contr
    


    
    
  
def get_norm(y_final,M,k,tabl_inter_bond,A_glabal,B_glabal,A_inter,B_inter,tabl_inter,tabl_bond,N2,N1):
    h1=1./(N1+1)
    h2=1./(N2+1)
    N_I=N1*N2
    N_T=(N1+2)*(N2+2)
    N_B=2*(N1+N2)+4

    y_Mplus1=y_final[:,-1]
    y_M=y_final[:,M]
    y_pri_Mplus1=(1./k)*(y_Mplus1-y_M)
    
    norm_y_Mplus1=np.sqrt(integral_phi_psi(tabl_inter_bond,A_glabal,N_T,y_Mplus1,y_Mplus1,h1,h2))
    print(norm_y_Mplus1)
    
    # norm of y^' at T in H^-1
    C_sm_l=second_member_lap(A_glabal,tabl_inter,tabl_bond,y_M,y_Mplus1,np.zeros(N_T),M,N_I,N_B,k)
    uu=Solver_lapl(A_inter,B_inter,C_sm_l,N_I,N_B,N2,N1,k,y_M,y_Mplus1,np.zeros(N_T))
    norm_y_Mplus1_prim=np.sqrt(integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,uu,uu,h1,h2))
    print(norm_y_Mplus1_prim)  
    
@njit
def boundary_norm(A_glabal,tabl_bond,V_h,M,k,N_B,h1,h2):
    Sum_b=0.
    l=N_B-2
    for i in range(l+1):
        for n in range(M+1):
           Sum_b=Sum_b+V_h[i,n]**2
                
    norm_b=np.sqrt((1./k)*(4./(l+1))*Sum_b )
    return norm_b
    
def get_normcontrol(vecphi0_hat,vecphi1_hat,nodes,tabl_bond,AB,A_inter,A_glabal,k,N1,N2,M): 
    N_I=N1*N2
    N_B=2*(N1+N2)+4
    N_T=(N1+2)*(N2+2)
    h1=1./(N1+1)
    h2=1./(N2+1)
    # Solve the ph_Hat system 
    C_sm=np.zeros((N_I,M))
    matphi_hat=solver_wave2D(AB,A_inter,k,vecphi0_hat[:N_I],vecphi1_hat[:N_I],C_sm,N_I,M,N2,N1,np.zeros((N_B,M+2))) 
    #Normal
    V_h=get_normal(matphi_hat,N1,N2,h1,h2,M)
    
    normL2=np.sqrt(h2*np.sum(V_h[:N2+1]**2)+h2*np.sum(V_h[N_B-N2:]**2)+h1*np.sum(V_h[N2+2:2:N_B-N2-1]**2)+h1*np.sum(V_h[N2+3:2:N_B-N2]**2))
    print(normL2)
