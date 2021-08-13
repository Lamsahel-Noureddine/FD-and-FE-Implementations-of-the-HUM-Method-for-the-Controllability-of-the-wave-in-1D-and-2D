# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:03:13 2021

@author: L
"""
import numpy as np
import scipy as sp
from scipy import linalg as lg

from solver1Dellfini import *
from Function_initial_Conditions import *
from visualisationF import *
from plot_inst import *
from Functions_use import *
from Functions_util import *
from plotcontrolsystem import *
import time
#space discretization 
N=300
h=1./(N+1)
x=np.linspace(0,1,N+2)




############# CFL condition  AND A0,B0 matrices  #############
A0=4*np.diag(np.ones(N))+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1)
B0=2*np.diag(np.ones(N))-np.diag(np.ones(N-1),1)-np.diag(np.ones(N-1),-1)

# CFL
C=0.01

#Time discretization 
T=2.2
k=h*C
M=int(T/k)-1
t=np.linspace(0,T,M+2)

#matrix A0B0 in the scheme 
A0B0=2*A0-6*(C**2)*B0
'''

  # # # # # # # # # # # # # # # # test of solver  # # # # # # # #

            # # # wave equation  # # # # # # # #


# Call function 
vj=v(x[1:-1])
wj=w(x[1:-1])
Un=u(t)



ph_appr=solver1Dellfini(vj,wj,C,k,M,Un,A0,A0B0)

phi_exact=phi_exact(x,t,N,M)




time_t=40
plot_t(time_t,k,x,ph_appr,phi_exact)
'''
'''
# integral approximation and  verification
vjj1=np.cos(x)
vjj2=np.exp(x)
intvv=appInte(vjj1,vjj2,h,N)
print("integral exact of v.v : %f"%(0.5*(  np.exp(1)*( np.cos(1)+np.sin(1) ) - 1 )  ))
print("integral approx of v.v : %f"%(intvv))
intDelta_vDelta_v=appintdelta(vjj1,vjj2,h,N)

print("integral exact of Delta.vDelta.v : %f"%(  0.5*(  -np.exp(1)*np.sin(1)+ np.exp(1)*np.cos(1) -1 )     )    )
print("integral approx of Delta.vDelta.v : %f"%(intDelta_vDelta_v))
'''
'''
t_0=M+1
VEC=ph_appr[:,t_0]-phi_exact[:,t_0]
er=erreurL2(h,VEC,N)
print("erreur at %d =%f"%(t_0,er))
errT=plot_ErreurL2(t,k,h,ph_appr,phi_exact,N,M)
print("erreur = %.20f"%(errT))
'''

'''
             # # # Dirichlet equation   # # # # # # # #
psi0=x
psi1=k*x+x
vecy1=3*x
approx_lapl=solve1d_Laplacian(psi0,psi1,h,k,vecy1,A0,B0,N)
exact_lapl=phi_exact_lapl(x)

  ########## ERREUR ############
def_exact_appr=approx_lapl-exact_lapl
erreur=erreurL2(h,def_exact_appr,N)
print(erreur)
###################

pl.plot(x,approx_lapl,'r--',x,exact_lapl,'g-.')
pl.legend(('numerical','exact '),loc="upper right")

pl.show()

'''






# # # # # # # # # # # # # # # # CAll of function of the Conjugate gradient method  # # # # # # # #
t1=time.time()
vecphi0_hat,vecphi1_hat=CG_alg(x,k,h,C,M,A0,B0,A0B0)
t2=time.time()
print('CPU=%f'%(t2-t1))
print('///////')
# # # # # # # # # # # # # # # # control of the wave  equation  # # # # # # # #

y_final=control_function(vecphi0_hat,vecphi1_hat,x,k,h,C,M,N,A0,A0B0)
         # plot and test
#plotControlfunc(y_final,x,t)
plotposVit(y_final,x,M,k)
  # norms
print('/////')
get_norm(y_final,M,N,h,k,A0,B0)
vcontrol=get_normcontrol(vecphi0_hat,vecphi1_hat,x,C,k,h,M,N,A0,A0B0)
















