# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 03:58:06 2021

@author: L
"""
import numpy as np
from mesh import *
from assembling_local import *
from assembling_global import *
from  Functions_util import *
from Solver2Dellfini import *
from initial_conditions import *
from  Functions_use import *
from Alg_GC import *

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D



N1=10
h1=1/(N1+1)
N2=N1
h2=h1

N_tho=(N1+1)*(N2+1)
N_I=N1*N2
N_T=(N1+2)*(N2+2)
N_B=2*(N1+N2)+4

print('Mesh:')
nodes,rects,Inods,tabl_inter,tabl_bond,tabl_inter_bond=mesh_function(h1,h2,N1,N2)
'''
print(nodes)
print('////////////////////////////////////////////')
print(rects)
print('///////////////')
print(Inods)
print('//////////////////////')
print(tabl_inter)
print(tabl_bond)
'''
print("matrices locals:")
A_local,B_local=matrix_local(h1,h2)
'''
print(A_local)
print(B_local)
print(np.sum(A_local.flatten())-h1*h2)
'''
print("Globals Matrices:")
A_glabal=np.zeros((N_T,N_T))
B_glabal=np.zeros((N_T,N_T))
A_glabal,B_glabal=matrix_global(A_local,B_local,N_tho,Inods,nodes,A_glabal,B_glabal)
'''
print(A_glabal)
print(B_glabal)
'''
'''
print("Internal matrices:")
A_inter=A_glabal[tabl_inter,:][:,tabl_inter]
B_inter=B_glabal[tabl_inter,:][:,tabl_inter] 
'''
'''
print(A_inter) 
print(B_inter)  
'''
#Time mesh
C_F=0.3
k=C_F*h1 
T=3.
M=int(T/k)-1

t=np.linspace(0,T,M+2)

AB=2*A_glabal[tabl_inter,:][:,tabl_inter]-3*(k**2)*B_glabal[tabl_inter,:][:,tabl_inter]


print(' ######################" Test of Wave equation #######################"')
#print(C_sm)    


'''
term_sec=2*np.ones(N_I)
term_sec[0]=50
term_sec[2]=0
term_sec[5]=-10
reslt=prod_matrix_AB_vect(AB,term_sec,N2,N1,N_I)
#print(reslt-np.dot(AB,term_sec))
'''
'''
Vi=v(tabl_inter, N_I,nodes)
Wi=w(tabl_inter, N_I,nodes)
Un=U(t,M,nodes,tabl_bond,N_B)
#print(Vi)
#print(Wi)

print("the approximation of  wave equation  :")
print("the second member of wave equation :")
C_sm=second_member(A_glabal,B_glabal,tabl_inter,tabl_bond,Un,M,N_I,N_B,k) 
phi_approx=solver_wave2D(AB,A_glabal[tabl_inter,:][:,tabl_inter],k,Vi,Wi,C_sm,N_I,M,N2,N1,Un)
print("the exact solution of wave equation   ")
phi_exact=exact_solution(t,M,nodes,N_T,tabl_inter_bond)

inst=M
print("PLot defference between the exact and the app   at instant :%d  "%inst)
def_phi=phi_approx[:,inst]-phi_exact[:,inst]
axe_def=range(N_T)
plt.plot(axe_def,def_phi)
plt.show()



# integral approximation and  verification
print("the approximation of  integrales :")
pHI=PHI_int(t,M,nodes,N_T,tabl_inter_bond)

intvv=integral_phi_psi(tabl_inter_bond,A_glabal,N_T,pHI,pHI,h1,h2)
vexact_1=1./9
print("integral exact of v.v : %f"%vexact_1)
print("integral approx of v.v : %f"%(intvv))


intDelta_vDelta_v=integraldelta_phi_psi(tabl_inter_bond,B_glabal,N_T,pHI,pHI,h1,h2)
vexact_2=2./3
print("integral exact of Delta.vDelta.v : %f"%vexact_2)
print("integral approx of Delta.vDelta.v : %f"%(intDelta_vDelta_v))


print('erreur between the exact solution and the approximation of tha wave equation: ')
err=ErreurExa_app(phi_approx,phi_exact,M,tabl_inter_bond,A_glabal,N_T,k,h1,h2)
print(err)
'''
'''
print("########################### Dirichlet equation ############ ")





print("the second member of Dirichlet equation :")
Y_vec=get_vectY(t,nodes,N_T,tabl_inter_bond)
Psi0=np.zeros(N_T)

Psi1=np.zeros(N_T)
for i in range(N_T):
    idof=tabl_inter_bond[i]
    X=nodes[idof]
    #Y_vec[i]=2*(X[0]*(X[0]-1) +X[1]*(X[1]-1))
    Psi1[i]=np.sin(np.pi*t[1])*np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
 
    
C_sm_l=second_member_lap(A_glabal,tabl_inter,tabl_bond,Psi0,Psi1,Y_vec,M,N_I,N_B,k)
print("the approximation of  Dirichlet equation  :")
phi_l_app=Solver_lapl(A_glabal[tabl_inter,:][:,tabl_inter],B_glabal[tabl_inter,:][:,tabl_inter],C_sm_l,N_I,N_B,N2,N1,k,Psi0,Psi1,Y_vec)
print("the exact solution  of  Dirichlet equation  :")
phi_l_exact=exact_dir_solution(t,nodes,N_T,tabl_inter_bond)
print("PLot defference between the exact and the app   ")
def_phi_l=phi_l_app-phi_l_exact
axe_def=range(N_T)
plt.plot(axe_def,def_phi_l)
plt.show()
print('erreur between the exact solution and the approximation of tha wave equation: ')
erreur=np.sqrt( integral_phi_psi(tabl_inter_bond,A_glabal,N_T,def_phi_l,def_phi_l,h1,h2)       )
print(erreur)
'''

'''
############################################# CG algorithm ################""

print('Test of get normal ')
PH=PH_func(t,M,nodes,N_T,tabl_inter_bond)

appnormal=get_normal(PH,N1,N2,h1,h2,M)
exact_normal=exact_normalPH(N1,N2,tabl_bond,M,t,nodes)
instt=M
deff_normal=exact_normal[:,instt]-appnormal[:,instt]
axe_def=range(N_B)
plt.plot(axe_def,deff_normal)
plt.show()



deff_normal=exact_normal[:,instt]
axe_def=range(N_B)
plt.plot(axe_def,deff_normal)
plt.show()

deff_normal=appnormal[:,instt]
axe_def=range(N_B)
plt.plot(axe_def,deff_normal)
plt.show()

er=boundary_norm(A_glabal,tabl_bond,exact_normal-appnormal,M,k,N_B,h1,h2)
print(er)
'''

vecphi0_hat,vecphi1_hat=CG_alg(nodes,tabl_inter_bond,tabl_bond,tabl_inter,AB,A_glabal[tabl_inter,:][:,tabl_inter],B_glabal[tabl_inter,:][:,tabl_inter],A_glabal,B_glabal,k,N1,N2,M)
print("\\\\\\\\\\\\\///")
y_final=control_function(vecphi0_hat,vecphi1_hat,nodes,tabl_bond,tabl_inter_bond,tabl_inter,AB,A_glabal[tabl_inter,:][:,tabl_inter],A_glabal,B_glabal,k,N1,N2,M)

get_norm(y_final,M,k,tabl_inter_bond,A_glabal,B_glabal,A_glabal[tabl_inter,:][:,tabl_inter],B_glabal[tabl_inter,:][:,tabl_inter],tabl_inter,tabl_bond,N2,N1)

get_normcontrol(vecphi0_hat,vecphi1_hat,nodes,tabl_bond,AB,A_glabal[tabl_inter,:][:,tabl_inter],A_glabal,k,N1,N2,M)







