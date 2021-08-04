# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 17:24:57 2021

@author:L
"""

import numpy as np
from numba import njit







# integral approximation
@njit
def integ_phi_psi(h1,h2,N1,N2,Psi,pHi):
    S=0
    for i in range(N1+1):
        for j in range(N2+1):
            S=S+Psi[i,j]*pHi[i,j]
    return h1*h2*S
    


def approximationDerPhi(N1,N2,h1,h2,PHI):
    deltaphi_x=PHI.copy()
    deltaphi_x[N1][:]=(1./h1)*(PHI[N1][:]-PHI[N1-1][:])
    for i in range(N1):
        deltaphi_x[i][:]=(1./h1)*(PHI[i+1][:]-PHI[i][:])
        
    deltaphi_y=PHI.copy()
    deltaphi_y[:][N2]=(1./h2)*(PHI[:][N2]-PHI[:][N2-1])
    for j in range(N2):
        deltaphi_y[:][j]=(1./h2)*(PHI[:][j+1]-PHI[:][j])

    return deltaphi_x,deltaphi_y

def integ_deltaphi_deltapsi(h1,h2,N1,N2,Psi,pHi):
    
    deltapHi_x,deltapHi_y=approximationDerPhi(N1,N2,h1,h2,pHi)
    deltaPsi_x,deltaPsi_y=approximationDerPhi(N1,N2,h1,h2,Psi)
    integ1=integ_phi_psi(h1,h2,N1,N2,deltapHi_x,deltaPsi_x)
    integ2=integ_phi_psi(h1,h2,N1,N2,deltapHi_y,deltaPsi_y)
    return integ1+integ2
    
    
def normL2_inst_t(h1,h2,N1,N2,PHI,t):
    
    PHI_t=PHI[t]
    normL2=integ_phi_psi(h1,h2,N1,N2,PHI_t,PHI_t)
    return normL2

def Erreur_L2(h1,h2,N1,N2,PHIexact,PHIappro,M,k): 
    norm_vect_diff=np.zeros(M+2)
    vect_diff=PHIexact-PHIappro
    for t in range(0,M+2): 
        norm_vect_diff[t]=normL2_inst_t(h1,h2,N1,N2,vect_diff,t)
    
    erreurL2=0
    for j in range(0,M+2):
        erreurL2=erreurL2+norm_vect_diff[j]
        
    ERR=np.sqrt(k*erreurL2)
    return ERR
        
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    