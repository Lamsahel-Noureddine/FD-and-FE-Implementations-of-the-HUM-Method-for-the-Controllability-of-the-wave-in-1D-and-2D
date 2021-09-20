# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 02:23:00 2021

@author: nour
"""
import numpy as np
from scipy import linalg as lg
d1=2*np.ones(3)
d2=(-1)*np.ones(2)
A=np.diag(d1)+np.diag(d2,-1)+np.diag(d2,1) # A is a symetric  positive deinete matrix
print('A=',A)
b=np.array([1,1,1])

xpython=lg.solve(A,b)    # x with python  
print('la solution avec python de Ax=b est :',xpython)
print('Erreur entre inv(A)b et la solution avec python est : ',np.dot(lg.inv(A),b)-xpython)
print("////////////////////////////////////////////////")



n=A.shape[0]          
x=np.ones(n)
r=b-np.dot(A,x)                    #r(x)=b-Ax
m=30                          # nombres  des iterations maximal 
beta= lg.norm(r)
V=np.zeros((n,m+1))
V[:,0]=(1/beta)*r
alphaT=np.zeros(m)
betaT=np.zeros(m+1)
for j in range(m):
  w=np.dot(A,V[:,j])-betaT[j]*V[:,j-1]
  alphaT[j]= np.sum(w*V[:,j])
  w=w-alphaT[j]*V[:,j]
  betaT[j+1]=lg.norm(w)
  if betaT[j+1]==0:
    break
  else:
    V[:,j+1]=(1/betaT[j+1])*w
Beta=betaT[1:m]
T=np.diag(alphaT)+np.diag(Beta,1)+np.diag(Beta,-1)
e=np.zeros(m)
e[0]=1

y=beta*np.dot(lg.inv(T),e)
Vm=V[:m,:m]
x=x+np.dot(Vm,y)
print("la méthode converge vers ",x)