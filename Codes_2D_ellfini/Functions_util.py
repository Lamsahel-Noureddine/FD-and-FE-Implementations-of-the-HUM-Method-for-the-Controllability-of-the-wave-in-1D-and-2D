# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 04:37:10 2021

@author: L
"""
import numpy as np
from numba import njit



########################### wave equation ##################
@njit
def U(t,M,nodes,tabl_bond,N_B):
    Un=np.zeros((N_B,M+2))
    for i in range(N_B):
        idof=tabl_bond[i]
        X=nodes[idof]
        for n in range(M+2):
           Un[i,n]=t[n]*X[0]*X[1]+X[0]+1.
           #Un[i,n]=0.

    return Un
# integral approximation and  verification
@njit
def PHI_int(t,M,nodes,N_T,tabl_inter_bond):
    pHI=np.zeros(N_T)
    for i in range(N_T):
        idof=tabl_inter_bond[i]
        X=nodes[idof]
        pHI[i]=X[0]*X[1]
    
    return pHI

@njit
def exact_solution(t,M,nodes,N_T,tabl_inter_bond):
    exact_sol=np.zeros((N_T,M+2))
    for i in range(N_T):
        idof=tabl_inter_bond[i]
        X=nodes[idof]
        for n in range(M+2):
            exact_sol[i,n]=t[n]*X[0]*X[1]+X[0]+1
            #exact_sol[i,n]=np.sqrt(2)*np.cos(  np.pi*np.sqrt(2)*( t[n]-(1/(4*np.sqrt(2)))    )  )*np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
            
    return exact_sol


################ Dirichlet #################"    
@njit
def get_vectY(t,nodes,N_T,tabl_inter_bond):
    Y_vec=np.zeros(N_T)
    for i in range(N_T):
        idof=tabl_inter_bond[i]
        X=nodes[idof]
        #Y_vec[i]=2*(X[0]*(X[0]-1) +X[1]*(X[1]-1))
        Y_vec[i]=-np.pi*np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
    return Y_vec  
    
@njit
def exact_dir_solution(t,nodes,N_T,tabl_inter_bond):
    exact_sol_l=np.zeros(N_T)
    for i in range(N_T):
        idof=tabl_inter_bond[i]
        X=nodes[idof]
        #exact_sol_l[i]=X[0]*X[1]*(X[0]-1)*(X[1]-1)
        exact_sol_l[i]=(1/np.pi)*np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
    return exact_sol_l     
    
@njit
def PH_func(t,M,nodes,N_T,tabl_inter_bond):
    exact_sol=np.zeros((N_T,M+2))
    for i in range(N_T):
        idof=tabl_inter_bond[i]
        X=nodes[idof]
        for n in range(M+2):
            #exact_sol[i,n]=t[n]*X[0]*X[1]+X[0]+1
            exact_sol[i,n]=np.sqrt(2)*np.cos(  np.pi*np.sqrt(2)*( t[n]-(1/(4*np.sqrt(2)))    )  )*np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
            
    return exact_sol    

@njit    
def exact_normalPH(N1,N2,tabl_bond,M,t,nodes):
    N_B=2*(N1+N2)+4
    PH_dervnorma_exact=np.zeros((N_B,M+2))
    for n in range(M+2):
        term_n=np.sqrt(2)*np.cos(  np.pi*np.sqrt(2)*( t[n]-(1/(4*np.sqrt(2)))    )  )
        for j in range(N2+2):
            jdof=tabl_bond[j]
            X1=nodes[jdof]
            PH_dervnorma_exact[j,n]=-term_n*np.pi*np.cos(np.pi*X1[0])*np.sin(np.pi*X1[1])
        for i in range(N1):
            idof=tabl_bond[N2+2+2*i]
            X=nodes[idof]
            PH_dervnorma_exact[N2+2+2*i,n]=-term_n*np.pi*np.sin(np.pi*X[0])*np.cos(np.pi*X[1])
            
            iidof=tabl_bond[N2+2+2*i+1]
            X2=nodes[iidof]
            PH_dervnorma_exact[N2+2+2*i+1,n]=term_n*np.pi*np.sin(np.pi*X2[0])*np.cos(np.pi*X2[1])
           

        for j in range(N2+2):
            jdof=tabl_bond[j+2*N1+N2+2]
            X3=nodes[jdof]
            PH_dervnorma_exact[j+2*N1+N2+2,n]=term_n*np.pi*np.cos(np.pi*X3[0])*np.sin(np.pi*X3[1])


      
    return PH_dervnorma_exact
    
    
    
    
    
    
    