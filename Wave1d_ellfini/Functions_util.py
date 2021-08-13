# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 09:20:37 2021

@author: L
"""

import numpy as np
from Functions_use import *


# integral approximation
def appInte(uu,vv,h,N):
    S1=(h/3.)*np.sum(uu[1:-1]*vv[1:-1])
    S1=S1+(h/3.)*np.sum(uu*vv)
    S2=0.
    for i in range(N+1):
        S2=S2+uu[i]*vv[i+1]+uu[i+1]*vv[i]

    fin_reslut=S1+(h/6)*S2
    return fin_reslut
    

def appintdelta(uu,vv,h,N):
    S1=(1./h)*np.sum(uu[1:-1]*vv[1:-1])
    S1=S1+(1./h)*np.sum(uu*vv)
    S2=0.
    for i in range(N+1):
        S2=S2+uu[i]*vv[i+1]+uu[i+1]*vv[i]

    fin_reslut=S1-(1./h)*S2
    return fin_reslut
   

def erreurL2(h,VEC,N):
    norm_Erreur=np.sqrt( appInte(VEC,VEC,h,N)  )
    return norm_Erreur


