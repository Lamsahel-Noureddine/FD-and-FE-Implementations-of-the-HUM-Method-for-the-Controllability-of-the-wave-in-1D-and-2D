# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 04:13:45 2021

@author: L
"""
from numba import njit
import numpy as np

############## test of solver_wave ##############
 
#The boundary condition at 1
@njit
def u(t,Y,M,N2,N1,X):
    sizt=M+2
    
    sizY=N2
    Un3=np.zeros((sizt,sizY))
    Un1=np.zeros((sizt,sizY))
    
    sizX=N1
    Un2=np.zeros((sizt,sizX))
    Un4=np.zeros((sizt,sizX))
    for n in range(sizt):
        for j in range(sizY):
            Un3[n,j]=t[n]*Y[j]+2.
            Un1[n,j]=1.
        for i in range(sizX):
            Un2[n,i]=X[i]+1.
            Un4[n,i]=t[n]*X[i]+X[i]+1.
            
            
    return Un1,Un2,Un3,Un4

    
@njit
def exactsolution(X,Y,t,N1,N2,M):
    excMAt_solution=np.zeros((M+2,N1,N2))
    for n in range(M+2):
        for i in range(N1):
            for j in range(N2):
                excMAt_solution[n,i,j]=t[n]*X[i]*Y[j]+X[i]+1
    return excMAt_solution