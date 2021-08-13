# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:00:28 2021

@author L
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from Functions_use import *
from Functions_util import *

def plot_t(time_t,k,x,ph_appr,phi_exact):
    
    etat_t=ph_appr[:,time_t]
    Y=phi_exact[:,time_t]
    X=x

    pl.plot(X,etat_t,'r--',X,Y,'g-.')
    pl.legend(('numerical','exact '),loc="upper right")
    # légende avec option loc
    # on met la valeur best pour loc si on laisse le dispositif
    # choisir le meilleur emplacement pour la légende
    pl.xlabel('x')
    pl.title('t=%f'%time_t)
    pl.show()
    

    
def plot_ErreurL2(t,k,h,ph_appr,phi_exact,N,M):
    
    Y=[]
    for tn in range(M+2):
        VEC=ph_appr[:,tn]-phi_exact[:,tn]
        yterm=erreurL2(h,VEC,N)
        Y.append(yterm)
    
    YY=np.array(Y)   
    # rectangles
    interreurL2rec=np.sqrt(k*np.sum(YY[:-1]**2))
    pl.plot(t,Y)
    pl.legend(' erreur ')
    pl.xlabel('t')
    pl.ylabel('ERReur(t)')
    pl.show()
    return interreurL2rec
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    