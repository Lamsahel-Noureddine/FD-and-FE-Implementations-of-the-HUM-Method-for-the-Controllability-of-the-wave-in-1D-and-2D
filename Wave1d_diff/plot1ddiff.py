# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:12:57 2021

@author: noureddine lamsahel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.colors as colors

N=1000
x,y,t,z=np.genfromtxt(r'tabvisualisation_1d_diff.dat',unpack=True)
xmin=x.min()
xmax=x.max()

ymin=y.min()
ymax=y.max()

vmin=z.min()
vmax=z.max()
xi=np.linspace(xmin,xmax,N)
yi=np.linspace(ymin,ymax,N)
zi=scipy.interpolate.griddata((x,y), z, (xi[None,:],yi[:,None]), method='linear')

def Custom_div_cmp(numcolors=11, name='Custom_div_cmp', mincol='blue', midcol='white', maxcol='red'):
    from matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.form_list(name=name, color=[mincol,midcol,maxcol], N=numcolors)
    return cmap
    

fig=plt.figure(figsize=None)
clvls=np.linspace(vmin,vmax,19)
a=plt.axes()
pc=a.contourf(xi,yi,zi,clvls, cmap='viridis')
pcs=a.contour(pc,levels=pc.levels[::2],linewidths=(0.5,),colors='w',extend='both',origin='lower')
a.clabel(pcs, fmt='%6.3f',fontsize=8,colors='w')
plt.colorbar(pc,ax=a)
a.axis([xmin,xmax,ymin,ymax])
plt.xlabel(r'$x$')    
plt.ylabel(r'$t$') 

#plt.title("analytical solution")
#plt.savefig('analytiqueSol.png', dpi=260 ,bbox_inches='tight')
plt.title(" numerical solution")  
plt.savefig('numericalSol.png', dpi=260 ,bbox_inches='tight')
plt.show()