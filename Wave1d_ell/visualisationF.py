# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:41:18 2021

@author: EL BERDAI ADAM
"""

import numpy as np
from Functions_use import *
def visuawav_app_ana_0(ph_appr,x,t):
    N=np.size(x)-2
    M=np.size(t)-2
    filevisua=open("tabvisualisation_1dwave_0.dat",'w')
    for n in range(0,M+2):
        for j in range(0,N+1):
            filevisua.write("%f  \t %f \t %f  \n"%(x[j],t[n],np.abs(phi_exact(x[j],t[n])-ph_appr[j,n])))
     
    filevisua.close()
    filevisua=open("tabvisualisation_1dwave_0.dat",'r')
    
    
