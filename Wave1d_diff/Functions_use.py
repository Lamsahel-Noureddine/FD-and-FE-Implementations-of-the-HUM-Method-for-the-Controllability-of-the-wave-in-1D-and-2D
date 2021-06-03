
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:38:39 2021

@author: noureddine lamsahel
"""
import numpy as np

############## test of solve1d_wave ##############
 
#The boundary condition at 1
def u(x):
    return 0.*x

#exact solution of the wave equation 
def phi_exact(x,t):
    
    return (1./2)*( np.sin(np.pi*(x+t)) + np.sin(np.pi*(x-t)) )


############## test of solve1d_Laplacian ##############

#exact solution of the laplacian  equation
def phi_exact_lapl(x):
    return (-1./2)*(x**2 -x )

