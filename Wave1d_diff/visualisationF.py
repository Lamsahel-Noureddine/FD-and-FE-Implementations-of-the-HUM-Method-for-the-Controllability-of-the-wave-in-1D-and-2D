# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:03:19 2021

@author: noureddine lmasahel
"""
import numpy as np
from Functions_use import *

def visua_app_ana(A_final,x,t):
    N=np.size(x)-2
    M=np.size(t)-2
    filevisua=open("tabvisualisation_1d_diff.dat",'w')
    for n in range(0,M+2):
        for j in range(0,N):
            filevisua.write("%f \t %f \t %f \t %f \n"%(x[j+1],t[n],phi_exact(x[j+1],t[n]),A_final[j,n]))
            
    filevisua.close()
    filevisua=open("tabvisualisation_1d_diff.dat",'r')