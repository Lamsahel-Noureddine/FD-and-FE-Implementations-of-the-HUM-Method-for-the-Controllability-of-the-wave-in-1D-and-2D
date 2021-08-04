# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:56:09 2021

@author: L
"""


from Solversdiff2D import *
from Functions_intial_conditions import *
from Functions_use import *
import numpy as np
from Function_util import *
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from plot import *
import time
#Space discretization 
  ##x
N1=80
h1=1./(N1+1)
x=np.linspace(0,1,N1+2)
  ###y
N2=N1  
y=x.copy() 
h2=h1
#Time discretization 
T=3                                                      #2.2*np.sqrt(2)
# CFL 
C1=1./np.sqrt(2)
C2=C1
k=h1*C1
M=int(T/k)-1
t=np.linspace(0,T,M+2)
print(C1)


'''
xx=x[1:-1]
yy=y[1:-1]
Xx=y0(xx,yy)
        
X, Y = np.meshgrid(xx, yy)
fig = plt.figure(figsize=(7, 5), dpi=100)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Xx, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#ax.set_zlim(1, 2.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('ph0')
'''
  # # # # # # # # # # # # # # # # test of solver  # # # # # # # #
'''
# Solve the wave system 
# Call function 
xx=x[1:-1]
yy=y[1:-1]

Vij=v(xx,yy,N1,N2)
Wij=w(xx,yy,N1,N2)
Un1,Un2,Un3,Un4=u(t,yy,M,N2,N1,xx)

# Solve the wave system


Phi_sol=Solver_Wave(M,N1,N2,k,C1,C2,Un1,Un2,Un3,Un4,Vij,Wij)
exact_ph=exactsolution(xx,yy,t,N1,N2,M)

tn=300
def_y=Phi_sol[tn]-exact_ph[tn]


X, Y = np.meshgrid(xx, yy)
fig = plt.figure(figsize=(7, 5), dpi=100)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, def_y, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#ax.set_zlim(1, 2.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$ t=%d$'%(tn))

err=Erreur_L2(h1,h2,N1,N2,exact_ph,Phi_sol,M,k)
print(err)
'''

'''
# solve the laplacian system
xx=x[1:-1]
yy=y[1:-1]
Y=np.zeros((N1,N2))
psi0=np.zeros((N1,N2))
psi1=np.zeros((N1,N2))
sol_exact=np.zeros((N1,N2))
for i in range(N1):
    for j in range(N2):
        Y[i,j]=2*(xx[i]*(xx[i]-1)+yy[j]*(yy[j]-1))
        sol_exact[i,j]=xx[i]*yy[j]*(xx[i]-1)*(yy[j]-1)
  
                        
appr=solve_Dirichlet(N1,N2,h1,h2,k,psi0,psi1,Y)

#plot
NN=N1*N2
X=range(NN)
Y1=appr.flatten()
Y2=sol_exact.flatten()
plt.plot(X,Y1-Y2)
'''
'''
# integral approximation and  verification
pHi=np.zeros((N1+1,N2+1))
for i in range(N1+1):
    for j in range(N2+1):
        pHi[i,j]=np.sin(np.pi*x[i])*np.sin(np.pi*y[j])
       # pHi[i,j]=x[i]+y[j]
        
        
intvv=integ_phi_psi(h1,h2,N1,N2,pHi,pHi)
print("integral exact of v.v : %f"%(1/4))
print("integral approx of v.v : %f"%(intvv))


intDelta_vDelta_v=integ_deltaphi_deltapsi(h1,h2,N1,N2,pHi,pHi)

print("integral exact of Delta.vDelta.v : %f"%(np.pi**2/2))
print("integral approx of Delta.vDelta.v : %f"%(intDelta_vDelta_v))
'''
# # # # # # # # # # # # # # # # CAll of function of the Conjugate gradient method  # # # # # # # #


t1=time.time()
vecphi0_hat,vecphi1_hat=CG_alg(x,y,k,h1,h2,C1,C2,M)
t2=time.time()
print('CPU=%f'%(t2-t1))
print('///////')

# # # # # # # # # # # # # # # # control of the wave  equation  # # # # # # # #

y_final=control_function(vecphi0_hat,vecphi1_hat,x,y,k,h1,h2,M,N1,N2,C1,C2)


         # plot and test
plotControlfunc(y_final,x,y,M)
#plotposVit(y_final,x,M,h)
  # norms

print('/////')
get_norm(y_final,M,N1,N2,h1,h2,k)
get_normcontrol(vecphi0_hat,vecphi1_hat,k,h1,h2,M,N1,N2,C1,C2)  

#v_h=get_normcontrol(vecphi0_hat,vecphi1_hat,x,r,k,h,M)











