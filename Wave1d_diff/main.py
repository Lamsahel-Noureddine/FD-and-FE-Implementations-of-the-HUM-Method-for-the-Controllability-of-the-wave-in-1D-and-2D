# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:12:47 2021

@author: noureddine lamsahel
"""
import numpy as np 
from Function_initial_Conditions import *
from solver1ddiff import *
from Functions_use import *
from visualisationF import *
from Functions_util import *
from plotcontrolsystem1ddiff import *
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
#Space discretization 
N=499
h=1./(N+1)
x=np.linspace(0,1,N+2)

#Time discretization 
T=2.2
# CFL 
r=1.
k=h*np.sqrt(r)
M=int(T/k -1)
t=np.linspace(0,T,M+2)




'''
  # # # # # # # # # # # # # # # # test of solver  # # # # # # # #
  
# Call function 
vj=v(x[1:-1])

wj=w(x[1:-1])
Un=u(t)

vecpsi0=np.zeros(N)
vecpsi1=np.zeros(N)
y11=np.ones(N)

# solve the laplacian system
phitest=solve1d_Laplacian(vecpsi0,vecpsi1,h,k,y11)
# visualisation and verification of solve1d_Laplacian


# Solve the wave system 
A_final=solve1d_wave(vj,wj,Un,r,k,M)
# visualisation and verification of solve1d_wave
visuawav_app_ana_0(A_final,x,t)
xx,tt,z=np.genfromtxt(r'tabvisualisation_1dwave_diff_0.dat',unpack=True)
nn=np.size(z)
XX=range(nn)
plt.plot(XX,z)

visuawav_app_ana(A_final,x,t)
visuawav_app_ana_0(A_final,x,t)
xx,tt,z=np.genfromtxt(r'tabvisualisation_1dwave_diff_0.dat',unpack=True)
nn=np.size(z)
XX=range(nn)
plt.plot(XX,z)
'''

'''
# integral approximation and  verification
vjj=v(x[:-1])
intvv=appInte(vjj,vjj,h)
print("integral exact of v.v : %f"%(1/2))
print("integral approx of v.v : %f"%(intvv))
intDelta_vDelta_v=appintdelta(vjj,vjj,h)

print("integral exact of Delta.vDelta.v : %f"%(np.pi**2/2))
print("integral approx of Delta.vDelta.v : %f"%(intDelta_vDelta_v))

# # # # # # # # # # # # # # # # CAll of function of the Conjugate gradient method  # # # # # # # #
'''

vecphi0_hat,vecphi1_hat=CG_alg(x,r,k,h,M)

# # # # # # # # # # # # # # # # control of the wave  equation  # # # # # # # #

y_final=control_function(vecphi0_hat,vecphi1_hat,x,r,k,h,M)
         # plot and test
#plotControlfunc(y_final,x,t)
#plotposVit(y_final,x,M,h)
  # norms

print('/////')
get_norm(y_final,M,N,h,k)

v_h=get_normcontrol(vecphi0_hat,vecphi1_hat,x,r,k,h,M)

'''
plt.plot(t,v_h)
plt.xlabel('$t$')
plt.ylabel('$v_h(t)$')

''' 

'''
yy=y0(x)
print(yy)
plt.plot(x,yy,'r.')
plt.xlabel('$x$')
plt.ylabel('$y^{0}(x)$')
'''

