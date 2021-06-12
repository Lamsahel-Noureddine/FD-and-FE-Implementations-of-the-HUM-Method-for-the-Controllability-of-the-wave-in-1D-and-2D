# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:03:13 2021

@author: EL BERDAI ADAM
"""
import numpy as np
import scipy as sp
from solver1Dellfini import *
from Function_initial_Conditions import *
from visualisationF import *
from plot_inst import *
from Functions_use import *

#space discretization 
N=100
h=1./(N+1)
x=np.linspace(0,1,N+2)

#Time discretization 
M=100
T=2.2
k=T/(M+1)
t=np.linspace(0,T,M+2)


  # # # # # # # # # # # # # # # # test of solver  # # # # # # # #
  
             # # # wave equation  # # # # # # # #
  
'''
# Call function 
vj=v(x[1:-1])
wj=w(x[1:-1])
Un=u(t)

#ph_appr=solver1Dellfini_band0(vj,wj,h,k,M)

ph_appr=solver1Dellfini(vj,wj,h,k,M,Un)
visuawav_app_ana_0(ph_appr,x,t)

time_t=1
plot_t(time_t,k,ph_appr,x)

'''
             # # # Dirichlet equation   # # # # # # # #
psi0=np.zeros(N)
psi1=np.zeros(N)
X=x[1:-1]
vecy1=np.cos(np.pi*X)
normalphi_t1=0.
normalphi_t0=0.
approx_lapl=solve1d_Laplacian(psi0,psi1,h,k,vecy1,normalphi_t1,normalphi_t0)

Y1=approx_lapl[1:]
Y2=phi_exact_lapl(X)
pl.plot(X,Y1,'r--',X,Y2,'g-.')
pl.legend(('numerical','exact '),loc="upper right")
pl.xlabel('x')
pl.show()



















