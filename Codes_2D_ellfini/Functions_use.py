# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 10:00:29 2021

@author: L
"""
import numpy as np
from numba import njit

@njit
def integral_phi_psi(tabl_inter_bond,A_glabal,N_T,pHi,pSi,h1,h2):
    sum_int=0.
    for i in range(N_T):
        idof=tabl_inter_bond[i]
        for j in range(N_T):
            jdof=tabl_inter_bond[j]
            sum_int=sum_int+pHi[i]*pSi[j]*(h1*h2/9)*A_glabal[idof,jdof]
    return sum_int
        
@njit
def integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,pHi,pSi,h1,h2):
    sum_int=0.
    for i in range(N_T):
        idof=tabl_inter_bond[i]
        for j in range(N_T):
            jdof=tabl_inter_bond[j]
            sum_int=sum_int+pHi[i]*pSi[j]*(h1*h2/3)*B_glabal[idof,jdof]
    return sum_int


@njit
def ErreurExa_app(phi_approx,phi_exact,M,tabl_inter_bond,A_glabal,N_T,k,h1,h2):
    Uex_app=[]
    for n in range(M+2):
        term_secend_err=phi_approx[:,n]-phi_exact[:,n]
        Uex_app.append(integral_phi_psi(tabl_inter_bond,A_glabal,N_T,term_secend_err,term_secend_err,h1,h2))
    sum_err=0.
    for n in range(M+1):
        sum_err=sum_err+k*Uex_app[n]
    erreur_total=np.sqrt(sum_err)
    return erreur_total
    
    
    