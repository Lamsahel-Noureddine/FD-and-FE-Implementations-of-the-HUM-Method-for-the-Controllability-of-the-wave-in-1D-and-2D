# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:21:28 2021

@author: noureddine lamsahel
"""
import numpy as np

############## test of solve1d_wave ##############

#The initial conditions 

def v(x):              # phi^0
    return np.sin(np.pi*x)

def w(x):          # phi^1
    return np.sqrt(2)*np.sin(np.pi*x)




