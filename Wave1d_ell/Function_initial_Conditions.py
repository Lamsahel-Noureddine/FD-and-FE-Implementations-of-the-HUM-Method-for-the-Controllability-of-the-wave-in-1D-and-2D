# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:31:50 2021

@author: EL BERDAI ADAM
"""

import numpy as np

############## test of solver1Dellfini ##############

#The initial conditions 

def v(x):              # phi^0
    return np.sin(2*np.pi*x)

def w(x):          # phi^1
    return 0.*x