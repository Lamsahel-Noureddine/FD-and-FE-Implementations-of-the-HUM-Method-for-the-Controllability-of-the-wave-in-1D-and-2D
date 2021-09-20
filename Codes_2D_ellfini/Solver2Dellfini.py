# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 05:13:02 2021

@author: L
"""
import numpy as np
from numba import njit
from scipy import linalg as lg
from scipy.sparse import linalg , coo_matrix
from initial_conditions import *

###########  Product Matrix-vector
@njit
def produc_AB12_vect(AB_1,sec_term_temp,N2):
    reslut=np.zeros(N2)
    lamda_1=AB_1[0,0]
    lamda_2=AB_1[0,1]
    reslut[0]=lamda_1*sec_term_temp[0]+(lamda_2)*sec_term_temp[1]
    for i in range(1,N2-1):
        reslut[i]=lamda_2*sec_term_temp[i-1]+lamda_1*sec_term_temp[i]+lamda_2*sec_term_temp[i+1]
    reslut[N2-1]=lamda_2*sec_term_temp[N2-2]+lamda_1*sec_term_temp[N2-1]
    return reslut

@njit    
def prod_matrix_AB_vect(AB,term_sec,N2,N1,N_I):
    AB_1=AB[:N2,:N2]
    AB_2=AB[:N2,N2:2*(N2)]
    result_prod=np.zeros(N_I)
    result_prod[:N2]=produc_AB12_vect(AB_1,term_sec[:N2],N2)+produc_AB12_vect(AB_2,term_sec[N2:N2*2],N2)
    for i in range(1,N1-1):
        result_prod[(i)*N2:(i+1)*N2]=produc_AB12_vect(AB_2,term_sec[(i-1)*N2:i*N2],N2)+produc_AB12_vect(AB_1,term_sec[(i)*N2:(i+1)*N2],N2)+produc_AB12_vect(AB_2,term_sec[(i+1)*N2:(i+2)*N2],N2)
        
    result_prod[(N1-1)*N2:]=produc_AB12_vect(AB_2,term_sec[(N1-2)*N2:(N1-1)*N2],N2)+produc_AB12_vect(AB_1,term_sec[(N1-1)*N2:(N1)*N2],N2)
    return result_prod


##################"


def solverlinear(matrA,vecb):
    cmatrA=matrA.copy()
    cvecb=vecb.copy()
    
    cmatrA=coo_matrix(cmatrA)
    cmatrA=cmatrA.tocsc()
    cmatrA=cmatrA.astype(np.float64)
    cvecb=cvecb.astype(np.float64)
    #C=dsolve.spsolve(A,b,use_umfpack=False)
    lu=linalg.splu(cmatrA)
    C_sol=lu.solve(cvecb)
    return C_sol

#solver_wave
def solver_wave2D(AB,A_inter,k,Vi,Wi,C_sm,N_I,M,N2,N1,Un):
    PHI=np.zeros((N_I,M+2))
    ph_0=Vi.copy()
    ph_1=ph_0+k*(Wi.copy())
    
    PHI[:,0]=ph_0
    PHI[:,1]=ph_1
    for n in range(1,M+1):
        termsc_wave=prod_matrix_AB_vect(AB,ph_1,N2,N1,N_I)-prod_matrix_AB_vect(A_inter,ph_0,N2,N1,N_I)+C_sm[:,n-1]
        ph=solverlinear(A_inter,termsc_wave)
        PHI[:,n+1]=ph
        ph_0=ph_1
        ph_1=ph
    PHI=np.vstack((PHI,Un))
    return PHI

            
#solver Dirichlet            
def Solver_lapl(A_inter,B_inter,C_sm_l,N_I,N_B,N2,N1,k,Psi0,Psi1,Y_vec):
    phi_l=np.zeros(N_I)
    Y_l=(1/k)*( Psi1[:N_I]- Psi0[:N_I] )-Y_vec[:N_I]
    AY_vec=prod_matrix_AB_vect(A_inter,Y_l,N2,N1,N_I)
    secterm_l=AY_vec+C_sm_l
    phi_l=solverlinear(3*B_inter,secterm_l)
    phi_l=np.hstack((phi_l,np.zeros(N_B)))
    return phi_l
    

           
    
    
    
    














