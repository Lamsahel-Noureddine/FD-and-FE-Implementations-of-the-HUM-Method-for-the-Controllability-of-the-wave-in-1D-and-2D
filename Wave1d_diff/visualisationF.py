# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:03:19 2021

@author: noureddine lmasahel
"""
import numpy as np
from Functions_use import *

def visuawav_app_ana(A_final,x,t):
    N=np.size(x)-2
    M=np.size(t)-2
    filevisua=open("tabvisualisation_1dwave_diff.dat",'w')
    for n in range(0,M+2):
        for j in range(0,N+1):
            filevisua.write("%f  \t %f \t %f \t %f \n"%(x[j],t[n],phi_exact(x[j],t[n]),A_final[j,n]))
            
    filevisua.close()
    filevisua=open("tabvisualisation_1dwave_diff.dat",'r')
    
def visuaLap_app_ana(phitest,x):
    N=np.size(x)-2
    filevisua=open("tabvisualisation_1dlap_diff.dat",'w')
    for j in range(1,N+1):
        filevisua.write("%f  \t %f \t %f  \n"%(x[j],phi_exact_lapl(x[j]),phitest[j]))
            
    filevisua.close()

    filevisua=open("tabvisualisation_1dlap_diff.dat",'r')



def visuawav_app_ana_0(A_final,x,t):
    N=np.size(x)-2
    M=np.size(t)-2
    filevisua_0=open("tabvisualisation_1dwave_diff_0.dat",'w')
    for n in range(0,M+2):
        for j in range(0,N+1):
            filevisua_0.write("%f  \t %f \t %f  \n"%(x[j],t[n],np.abs(phi_exact(x[j],t[n])-A_final[j,n])))
            
    filevisua_0.close()
    filevisua_0=open("tabvisualisation_1dwave_diff_0.dat",'r')
