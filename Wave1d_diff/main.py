# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:12:47 2021

@author: noureddine lamsahel
"""
import numpy as np 
from Function_initial_Conditions import *
from solver1ddiff import *
from Functions_use import *
from visualisationF import *
from Functions_util import *
#Space discretization 
N=200
h=1/(N+1)
x=np.linspace(0,1,N+2)

#Time discretization 
T=2.5
M=1000
k=T/(M+1)
t=np.linspace(0,T,M+2)

# Coefficientin the numerical schema 
r=(k/h)**2


  # # # # # # # # # # # # # # # # test of solver  # # # # # # # #
  
# Call function 
vj=v(x[1:-1])
wj=w(x[1:-1])
Un=u(t)

psi0=np.zeros(N)
psi1=np.zeros(N)
y11=np.ones(N)

# solve the laplacian system
phitest=solve1d_Laplacian(psi0,psi1,h,k,y11)
# visualisation and verification of solve1d_Laplacian

visuaLap_app_ana(phitest,x)

# Solve the wave system 
A_final=solve1d_wave(vj,wj,Un,r,k,M)
# visualisation and verification of solve1d_wave
visuawav_app_ana(A_final,x,t)

# integral approximation and  verification
vjj=v(x[:-1])
intvv=appInte(vjj,vjj,h)
print("integral exact of v.v : %f"%(1/2))
print("integral approx of v.v : %f"%(intvv))
intDelta_vDelta_v=appintdelta(vjj,vjj,h)

print("integral exact of Delta.vDelta.v : %f"%(np.pi**2/2))
print("integral approx of Delta.vDelta.v : %f"%(intDelta_vDelta_v))
