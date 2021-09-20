# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 09:49:57 2021

@author: L
"""
import numpy as np
from numba import njit

#matrices locals
@njit
def matrix_local(h1,h2):
    A_local=np.zeros((4,4))
    B_local=np.zeros((4,4))
    
    a1_local=np.ones(4)
    a2_local=(1./2)*np.ones(3)
    a3_local=(1./4)*np.ones(2)
    a4_local=(1./2)**np.ones(1)
    A_local=np.diag(a1_local)+np.diag(a2_local,1)+np.diag(a3_local,2)+np.diag(a4_local,3)+np.diag(a2_local,-1)+np.diag(a3_local,-2)+np.diag(a4_local,-3)
    
    b1_local=  (  (1/(h1**2))+  (1/(h2**2))   )  *np.ones(4)
    
    b2_local=np.ones(3)
    b2_local[0]=(  (  (-1/(h1**2))+  (1/(2*(h2**2)))   )  )
    b2_local[1]=(  (  (1/(2*(h1**2)))+  (-1/(h2**2))   )  )
    b2_local[2]=( (  (-1/(h1**2))+  (1/(2*(h2**2)))   )  )
    b3_local=(  (1./2 )*(  (-1/(h1**2))+  (-1/(h2**2))   )  )*np.ones(2)
    b4_local=(  (  (1/(2*(h1**2)))+  (-1/(h2**2))   )  )*np.ones(1)
    B_local=np.diag(b1_local)+np.diag(b2_local,1)+np.diag(b3_local,2)+np.diag(b4_local,3)+np.diag(b2_local,-1)+np.diag(b3_local,-2)+np.diag(b4_local,-3)
    return A_local,B_local

