# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:19:38 2021

@author: nour
"""

import numpy as np
from numba import njit

@njit
def matrix_global(A_local,B_local,N_tho,Inods,nodes,A,B):
    for r in range(N_tho):
        global_inds=Inods[r]
        for i in range(4):
            for j in range(4):
                i_dof=global_inds[i]
                j_dof=global_inds[j]
                A[i_dof,j_dof]=A[i_dof,j_dof]+A_local[i,j]
                B[i_dof,j_dof]=B[i_dof,j_dof]+B_local[i,j]
   
    return A,B

###############"second members ############"
@njit
def second_member(A_glabal,B_glabal,tabl_inter,tabl_bond,Un,M,N_I,N_B,k):
    C_sm=np.zeros((N_I,M))
    for i in range(N_I):
        i_dof=tabl_inter[i]
        for n in range(1,M+1):
            term_tem=0.
            for j in range(N_B):
                j_dof=tabl_bond[j]
                term_tem=term_tem+((Un[j,n+1]-2*Un[j,n]+Un[j,n-1]))*A_glabal[i_dof,j_dof]+3*(k**2)*Un[j,n]*B_glabal[i_dof,j_dof]
            C_sm[i,n-1]=-term_tem
          
                
            
    return C_sm  
    
@njit
def second_member_lap(A_glabal,tabl_inter,tabl_bond,Psi0,Psi1,Y_vec,M,N_I,N_B,k):
    C_sm_l=np.zeros(N_I)
    for i in range(N_I):
        i_dof=tabl_inter[i]
        term_tem=0.
        for j in range(N_B):
            j_dof=tabl_bond[j]
            term_tem=term_tem+((1/k)*(Psi1[N_I+j]-Psi0[N_I+j])-Y_vec[N_I+j]  )*A_glabal[i_dof,j_dof]
        C_sm_l[i]=term_tem
          
                
            
    return C_sm_l  