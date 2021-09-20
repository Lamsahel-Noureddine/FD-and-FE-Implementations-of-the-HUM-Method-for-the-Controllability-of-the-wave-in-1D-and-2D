# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 06:19:33 2021

@author: L
"""
import numpy as np
from numba import njit

############## test of solver_wave ##############
@njit
def v(tabl_inter, N_I,nodes):
    Vj=np.zeros(N_I)
    for i in range(N_I):
        idof=tabl_inter[i]
        X=nodes[idof]
        Vj[i]=X[0]+1
        #Vj[i]=np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
    return Vj   
  
@njit
def w(tabl_inter, N_I,nodes):
    Wj=np.zeros(N_I)
    for i in range(N_I):
        idof=tabl_inter[i]
        X=nodes[idof]
        Wj[i]=X[0]*X[1]
        #Wj[i]=np.pi*np.sqrt(2)*np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
    return Wj 



############## conjugate gradient  ##############

#The initial conditions 
@njit
def y0(nodes,N_T,tabl_inter_bond): 
    Y0_vec=np.zeros(N_T)
    rhoo=0.2
    for i in range(N_T):
       idof= tabl_inter_bond[i]
       X=nodes[idof]
       r=(X[0]-0.35)**2+(X[1]-0.35)**2
       Y0_vec[i]=np.exp(-5*( r/rhoo**2)**6   )
       #Y0_vec[i]=X[0]*X[1]
       
    return  Y0_vec
    
          
  
 

@njit    
def y1(nodes,N_T,tabl_inter_bond):
    Y1_vec=np.zeros(N_T)
    for i in range(N_T):
       idof= tabl_inter_bond[i]
       X=nodes[idof]
       Y1_vec[i]=0.
       
    return  Y1_vec
    
#initialization 
@njit
def ph0(nodes,N_T,tabl_inter_bond):              # phi^0
    ph0_vec=np.zeros(N_T)
    for i in range(N_T):
       idof= tabl_inter_bond[i]
       X=nodes[idof]
       ph0_vec[i]=0.
       
    return  ph0_vec      #0
@njit
def ph1(nodes,N_T,tabl_inter_bond):          # phi^1
    ph1_vec=np.zeros(N_T)
    for i in range(N_T):
       idof= tabl_inter_bond[i]
       X=nodes[idof]
       ph1_vec[i]=0.
       
    return  ph1_vec      #0