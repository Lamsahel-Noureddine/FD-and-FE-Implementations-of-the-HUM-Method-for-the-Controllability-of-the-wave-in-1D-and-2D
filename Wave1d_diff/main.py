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

#Space discretization 
N=99
h=1/(N+1)
x=np.linspace(0,1,N+2)

#Time discretization 
T=2.5
M=499
k=T/(M+1)
t=np.linspace(0,T,M+2)
# Coefficientin the numerical schema 
r=(k/h)**2

# Call function 
vj=v(x[1:-1])
wj=w(x[1:-1])
Un=u(t)

# Solve the system
A_final=solve1d(vj,wj,Un,r,k,M)
# visualisation 
visua_app_ana(A_final,x,t)
