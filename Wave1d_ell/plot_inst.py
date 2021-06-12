# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:00:28 2021

@author: EL BERDAI ADAM
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from Functions_use import *


def plot_t(time_t,k,ph_appr,x):
    n=int(time_t/k)
    etat_t=ph_appr[:,n]
    
    xx=x[1:]
    Y=[]
    for ter in xx:
        Yind=phi_exact(ter,time_t)
        Y.append(Yind)

    pl.plot(xx,etat_t,'r--',xx,Y,'g-.')
    pl.legend(('numerical','exact '),loc="upper right")
    # légende avec option loc
    # on met la valeur best pour loc si on laisse le dispositif
    # choisir le meilleur emplacement pour la légende
    pl.xlabel('x')
    pl.title('t=%f'%time_t)
    pl.show()
    
