# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 03:58:59 2021

@author: L
"""
import numpy as np
from numba import njit

@njit
def mesh_function(h1,h2,N1,N2):
    nodes=[]
    rects=[]
    Inods=[]
    tabl_inter=[]
    tabl_bond=[]
    for i in range(N1+2):
        for j in range(N2+2):
            nodes.append([i*h1,j*h2])
           
                
            
    
    for i in range(N1+1):
        for j in range(N2+1):
            rects.append([nodes[i*(N2+2)+j],nodes[(i+1)*(N2+2)+j], nodes[(i+1)*(N2+2)+j+1],nodes[i*(N2+2)+j+1]  ])
            Inods.append([i*(N2+2)+j,(i+1)*(N2+2)+j,(i+1)*(N2+2)+j+1,i*(N2+2)+j+1])
            
    
    for i in range(1,N1+1):
        for j in range(1,N2+1):
            tabl_inter.append(i*(N2+2)+j)
    
    for j in range(N2+2):
        tabl_bond.append(j)
    for i in range(1,N1+1):
        tabl_bond.append(i*(N2+2))
        tabl_bond.append(i*(N2+2)+N2+1)
    for j in range(N2+2):
        tabl_bond.append((N1+1)*(N2+2)+j)
    
    tabl_inter_bond=tabl_inter+tabl_bond
    return np.array(nodes),rects,np.array(Inods),np.array(tabl_inter),np.array(tabl_bond),np.array(tabl_inter_bond)
    
