# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 01:57:38 2021

@author: L
"""

from numba import njit
import numpy as np
from scipy import linalg as lg
from scipy.sparse import linalg , coo_matrix
from Functions_intial_conditions import *
from Function_util import *
'''
@njit
def Solver_Wave(M,N1,N2,k,C1,C2,Un,Vij,Wij):
    Ph_0=Vij.copy()
    Ph_1=Ph_0+k*Wij.copy()
    
    Phi_sol=np.zeros((M+2,N1,N2))
    Phi_sol[0]=Ph_0
    Phi_sol[1]=Ph_1
    
    for n in range(1,M+1):
        ###### i=1 ####
         #j=1
        term1_01=2*Phi_sol[n][0][0]-Phi_sol[n-1][0][0]+C1*(Phi_sol[n][1][0]-2*Phi_sol[n][0][0])
        term2_02=C2*(Phi_sol[n][0][1]-2*Phi_sol[n][0][0])
        Phi_sol[n+1][0][0]=term1_01+term2_02
         #j=N2
        term1_N23=2*Phi_sol[n][0][-1]-Phi_sol[n-1][0][-1]+C1*(Phi_sol[n][1][-1]-2*Phi_sol[n][0][-1])
        term2_N24=C2*(-2*Phi_sol[n][0][-1]+Phi_sol[n][0][N2-2])
        Phi_sol[n+1][0][-1]=term1_N23+term2_N24
         # 1<j<N2
        for ind1 in range(1,N2-1):
            term1_ind15=2*Phi_sol[n][0][ind1]-Phi_sol[n-1][0][ind1]+C1*(Phi_sol[n][1][ind1]-2*Phi_sol[n][0][ind1])
            term2_ind16=C2*(Phi_sol[n][0][ind1+1]-2*Phi_sol[n][0][ind1]+Phi_sol[n][0][ind1-1])
            Phi_sol[n+1][0][ind1]=term1_ind15+term2_ind16
        
        ###### i=N1 ####
         #j=1
        term1_07=2*Phi_sol[n][-1][0]-Phi_sol[n-1][-1][0]+C1*(Un[0,n]-2*Phi_sol[n][-1][0]+Phi_sol[n][N1-2][0])
        term2_08=C2*(Phi_sol[n][-1][1]-2*Phi_sol[n][-1][0])
        Phi_sol[n+1][-1][0]=term1_07+term2_08
         #j=N2
        term1_N29=2*Phi_sol[n][-1][-1]-Phi_sol[n-1][-1][-1]+C1*(Un[-1,n]-2*Phi_sol[n][-1][-1]+Phi_sol[n][N1-2][-1])
        term2_N210=C2*(-2*Phi_sol[n][-1][-1]+Phi_sol[n][-1][N2-2])
        Phi_sol[n+1][-1][-1]=term1_N29+term2_N210
         # 1<j<N2
        for ind2 in range(1,N2-1):
            term1_ind211=2*Phi_sol[n][-1][ind2]-Phi_sol[n-1][-1][ind2]+C1*(Un[ind2,n]-2*Phi_sol[n][-1][ind2]+Phi_sol[n][N1-2][ind2])
            term2_ind212=C2*(Phi_sol[n][-1][ind2+1]-2*Phi_sol[n][-1][ind2]+Phi_sol[n][-1][ind1-1])
            Phi_sol[n+1][-1][ind2]=term1_ind211+term2_ind212
        
        #############################################
          ###### j=1 ####
         #i=1
        term1_0j13=2*Phi_sol[n][0][0]-Phi_sol[n-1][0][0]+C1*(Phi_sol[n][1][0]-2*Phi_sol[n][0][0])
        term2_0j14=C2*(Phi_sol[n][0][1]-2*Phi_sol[n][0][0])
        Phi_sol[n+1][0][0]=term1_0j13+term2_0j14
         #i=N1
        term1_N2j15=2*Phi_sol[n][-1][0]-Phi_sol[n-1][-1][0]+C1*(Un[0,n]-2*Phi_sol[n][-1][0]+Phi_sol[n][N1-2][0])
        term2_N2j16=C2*(Phi_sol[n][-1][1]-2*Phi_sol[n][-1][0])
        Phi_sol[n+1][-1][0]=term1_N2j15+term2_N2j16
         # 1<i<N1
        for ind3 in range(1,N1-1):
            term1_ind117=2*Phi_sol[n][ind3][0]-Phi_sol[n-1][ind3][0]+C1*(Phi_sol[n][ind3+1][0]-2*Phi_sol[n][ind3][0]+Phi_sol[n][ind3-1][0])
            term2_ind118=C2*(Phi_sol[n][ind3][1]-2*Phi_sol[n][ind3][0])
            Phi_sol[n+1][ind3][0]=term1_ind117+term2_ind118
        
        ###### j=N2 ####
         #i=1
        term1_019=2*Phi_sol[n][0][-1]-Phi_sol[n-1][0][-1]+C1*(Phi_sol[n][1][-1]-2*Phi_sol[n][0][-1])
        term2_020=C2*(-2*Phi_sol[n][0][-1]+Phi_sol[n][0][N2-2])
        Phi_sol[n+1][0][-1]=term1_019+term2_020
         #i=N1
        term1_N221=2*Phi_sol[n][-1][-1]-Phi_sol[n-1][-1][-1]+C1*(Un[-1,n]-2*Phi_sol[n][-1][-1]+Phi_sol[n][N1-2][-1])
        term2_N222=C2*(-2*Phi_sol[n][-1][-1]+Phi_sol[n][-1][N2-2])
        Phi_sol[n+1][-1][-1]=term1_N221+term2_N222
         # 1<i<N1
        for ind4 in range(1,N1-1):
            term1_ind223=2*Phi_sol[n][ind4][-1]-Phi_sol[n-1][ind4][-1]+C1*(Phi_sol[n][ind4+1][-1]-2*Phi_sol[n][ind4][-1]+Phi_sol[n][ind4-1][-1])
            term2_ind224=C2*(-2*Phi_sol[n][ind4][-1]+Phi_sol[n][ind4][N2-2])
            Phi_sol[n+1][ind4][-1]=term1_ind223+term2_ind224
            
        ###########################################
         ####### 1<i<N1 and 1<j<N2
        for ii in range(1,N1-1):
            for jj in range(1,N2-1):
                ter1=2*Phi_sol[n][ii][jj]-Phi_sol[n-1][ii][jj]+C1*(Phi_sol[n][ii+1][jj]-2*Phi_sol[n][ii][jj]+Phi_sol[n][ii-1][jj])
                ter2=C2*(Phi_sol[n][ii][jj+1]-2*Phi_sol[n][ii][jj]+Phi_sol[n][ii][jj-1])
                Phi_sol[n+1][ii][jj]=ter1+ter2
                
             
    return  Phi_sol      
'''
@njit    
def getMatrix_Wave(N1,N2,C1,C2):
    
    NN=N1*N2
    matrix_wave=np.zeros((NN,NN))
    lamda_scal=2*(1-C1**2-C2**2)
    vecA1=lamda_scal*np.ones(N2)
    vecA12=(C2**2)*np.ones(N2-1)
    A1=np.diag(vecA1)+np.diag(vecA12,-1)+np.diag(vecA12,1)
    
    A2=np.diag((C1**2)*np.ones(N2))
    for i in range(N1):
        for ii in range(N2):
            for jj in range(N2):
                matrix_wave[i*N2+ii,i*N2+jj]=A1[ii,jj]
        for j in range(N1):
            if np.abs(i-j)==1:
                for iik in range(N2):
                    for jjk in range(N2):
                        matrix_wave[i*N2+iik,j*N2+jjk]=A2[iik,jjk]
                    
      
    return   matrix_wave 

@njit
def prod_matrix_vectA(lamda,C2,sec_term_temp,N2):
    reslut=np.zeros(N2)
    reslut[0]=lamda*sec_term_temp[0]+(C2**2)*sec_term_temp[1]
    for i in range(1,N2-1):
        reslut[i]=(C2**2)*sec_term_temp[i-1]+lamda*sec_term_temp[i]+(C2**2)*sec_term_temp[i+1]
    reslut[N2-1]=(C2**2)*sec_term_temp[N2-2]+lamda*sec_term_temp[N2-1]
    return reslut


@njit    
def prod_matrix_wave_vect(C2,C1,term_sec,N2,N1):
    NN=N1*N2
    lamda_scal=2*(1-C1**2-C2**2)
    result_prod=np.zeros(NN)
    result_prod[:N2]=prod_matrix_vectA(lamda_scal,C2,term_sec[:N2],N2)+(C1**2)*term_sec[N2:N2*2]
    for i in range(1,N1-1):
        result_prod[(i)*N2:(i+1)*N2]=(C1**2)*term_sec[(i-1)*N2:i*N2]+prod_matrix_vectA(lamda_scal,C2,term_sec[(i)*N2:(i+1)*N2],N2)+(C1**2)*term_sec[(i+1)*N2:(i+2)*N2]
        
    
    result_prod[(N1-1)*N2:]=(C1**2)*term_sec[(N1-2)*N2:(N1-1)*N2]+prod_matrix_vectA(lamda_scal,C2,term_sec[(N1-1)*N2:(N1)*N2],N2)
    return result_prod

    
    
def Solver_Wave(M,N1,N2,k,C1,C2,Un1,Un2,Un3,Un4,Vij,Wij):
    NN=N1*N2
    
    Ph_0=Vij.copy()
    Ph_1=Ph_0+k*(Wij.copy())
    
    Phi_sol=np.zeros((M+2,N1,N2))
    Phi_sol[0]=Ph_0
    Phi_sol[1]=Ph_1

    phi0_flatt=Ph_0.flatten()
    phi1_flatt=Ph_1.flatten()
    
    
    for n in range(1,M+1):
        b_wave=np.zeros(NN)
        interm_b_wave2=np.zeros(NN)
        interm_b_wave4=np.zeros(NN)
        
        interm_b_wave1=(C1**2)*Un1[n,:]
        b_wave[:N2]=interm_b_wave1
        
        interm_b_wave3=(C1**2)*Un3[n,:]
        b_wave[NN-N2:]=interm_b_wave3
        for k in range(N1):
            interm_b_wave2[k*N2]=Un2[n,k]
            interm_b_wave4[(k+1)*N2-1]=Un4[n,k]
        
        b_wave=b_wave+(C2**2)*interm_b_wave2+(C2**2)*interm_b_wave4
    
        phi_iter=prod_matrix_wave_vect(C2,C1,phi1_flatt,N2,N1)-phi0_flatt+ b_wave

        phi_iter_resize=phi_iter.copy()
        phi_iter_resize.resize((N1,N2))
        Phi_sol[n+1]=phi_iter_resize

        phi0_flatt=phi1_flatt
        phi1_flatt=phi_iter
        
    return Phi_sol
    
@njit    
def getMatrix_Dirichlet(N1,N2,h1,h2):
    NN=N1*N2
    matrix_Direchlet=np.zeros((NN,NN))
    vecA1=(-2./(h1**2)-2./(h2**2))*np.ones(N2)
    vecA12=(1./(h2**2))*np.ones(N2-1)
    A1=np.diag(vecA1)+np.diag(vecA12,-1)+np.diag(vecA12,1)
    
    A2=np.diag((1./(h1**2))*np.ones(N2))
    for i in range(N1):
        for ii in range(N2):
            for jj in range(N2):
                matrix_Direchlet[i*N2+ii,i*N2+jj]=A1[ii,jj]
        for j in range(N1):
            if np.abs(i-j)==1:
                for iik in range(N2):
                    for jjk in range(N2):
                        matrix_Direchlet[i*N2+iik,j*N2+jjk]=A2[iik,jjk]
                    
      
    return   matrix_Direchlet                  
                        
                        

def solverlinear(matrA,vecb):
    matrixA=matrA.copy()
    vectb=vecb.copy()
    matrixA=coo_matrix(matrixA)
    matrixA=matrixA.tocsc()
    matrixA=matrixA.astype(np.float64)
    vectb=vectb.astype(np.float64)
    #C=dsolve.spsolve(A,b,use_umfpack=False)
    lu=linalg.splu(matrixA)
    C_sol=lu.solve(vectb)
    return C_sol


                      
def solve_Dirichlet(N1,N2,h1,h2,k,psi0,psi1,Y):
   matrix_Direchlet=getMatrix_Dirichlet(N1,N2,h1,h2)
   scendM=Y-(1./k)*(psi1-psi0)
   scendM_f=scendM.flatten()
   solutionapp=solverlinear(matrix_Direchlet,scendM_f)
   solutionapp.resize((N1,N2))
   return solutionapp
    


def addzerosVh(A,N1,N2):
    Acop=A.copy()
    AAcop=np.vstack((np.zeros(N2),Acop))
    AAAcop=np.hstack((np.zeros(N1+1)[:,np.newaxis],AAcop))
    return AAAcop











    
def CG_alg(x,y,k,h1,h2,C1,C2,M):
    N1=np.size(x)-2
    N2=np.size(y)-2
    iterMAx=100000
    eps=1e-10
    #initialization
    vecphi0=ph0(x[:-1],y[:-1])
    vecphi1=ph1(x[:-1],y[:-1])

  # # # #  #  iteration 0  # # # # # # #
    #solve ph_0 system  
    matphi=Solver_Wave(M,N1,N2,k,C1,C2,np.zeros((M+2,N2)),np.zeros((M+2,N1)),np.zeros((M+2,N2)),np.zeros((M+2,N1)),vecphi0[1:,1:],vecphi1[1:,1:])
    #solve psi_0 system 
    normalphi_1=(-1./h1)*matphi[:,0,:]
    normalphi_3=(-1./h1)*matphi[:,-1,:]
    
    normalphi_2=(-1./h2)*matphi[:,:,0]
    normalphi_4=(-1./h2)*matphi[:,:,-1]
    
    matpsi_int=Solver_Wave(M,N1,N2,k,C1,C2,normalphi_1[::-1],normalphi_2[::-1],normalphi_3[::-1],normalphi_4[::-1],np.zeros((N1,N2)),np.zeros((N1,N2)))    
    matpsi=matpsi_int[::-1]  
    
     #solve laplacian system
    vecy1=y1(x[1:-1],y[1:-1])
    vecphi0_tilde1=solve_Dirichlet(N1,N2,h1,h2,k,matpsi[0],matpsi[1],vecy1)
    vecphi0_tilde=addzerosVh(vecphi0_tilde1,N1,N2)
    
    matpsi_0add=addzerosVh(matpsi[0],N1,N2)
    vecphi1_tilde=y0(x[:-1],y[:-1])-matpsi_0add
    
    cond_t2=np.sqrt(integ_deltaphi_deltapsi(h1,h2,N1,N2,vecphi0_tilde,vecphi0_tilde)+integ_phi_psi(h1,h2,N1,N2,vecphi1_tilde,vecphi1_tilde))
     #stopping criteria
    t1=cond_t2.copy()
    t2=np.sqrt(integ_deltaphi_deltapsi(h1,h2,N1,N2,vecphi0,vecphi0)+integ_phi_psi(h1,h2,N1,N2,vecphi1,vecphi1))
    if t2==0.:
        t2=1.
    testcondition=t1/t2
    
    vecphi0_check=vecphi0_tilde.copy()
    vecphi1_check=vecphi1_tilde.copy()
   
    
    #  Descent
    itern=1
    print('yo bro ')
    while itern<=iterMAx and  testcondition > eps:
        print(itern)
        #solve ph_0_check system 
        matphi_check=Solver_Wave(M,N1,N2,k,C1,C2,np.zeros((M+2,N2)),np.zeros((M+2,N1)),np.zeros((M+2,N2)),np.zeros((M+2,N1)),vecphi0_check[1:,1:],vecphi1_check[1:,1:])
        #solve psi_0 system 
        normalphi_1_check_1=(-1./h1)*matphi_check[:,0,:]
        normalphi_1_check_3=(-1./h1)*matphi_check[:,-1,:]
        
        normalphi_1_check_2=(-1./h2)*matphi_check[:,0,:]
        normalphi_1_check_4=(-1./h2)*matphi_check[:,-1,:]
        matpsi_check_int=Solver_Wave(M,N1,N2,k,C1,C2,normalphi_1_check_1[::-1],normalphi_1_check_2[::-1],normalphi_1_check_3[::-1],normalphi_1_check_4[::-1],np.zeros((N1,N2)),np.zeros((N1,N2)))  
        matpsi_check=matpsi_check_int[::-1] 
        #solve laplacian system
        vecphi0_line1=solve_Dirichlet(N1,N2,h1,h2,k,matpsi_check[0],matpsi_check[1],np.zeros((N1,N2)))
        vecphi0_line=addzerosVh(vecphi0_line1,N1,N2)
        
        matpsi_0add_check=addzerosVh(matpsi_check[0],N1,N2)
        
        vecphi1_line=-matpsi_0add_check
        # calcul of rho_n

        term1=integ_deltaphi_deltapsi(h1,h2,N1,N2,vecphi0_tilde,vecphi0_tilde)+integ_phi_psi(h1,h2,N1,N2,vecphi1_tilde,vecphi1_tilde)
        term2=integ_deltaphi_deltapsi(h1,h2,N1,N2,vecphi0_line,vecphi0_check)+integ_phi_psi(h1,h2,N1,N2,vecphi1_line,vecphi1_check)
        
        rho=term1/term2
        #  #  #  #  go to n+1  #  #  #  # 
        vecphi0=vecphi0-rho*vecphi0_check
        vecphi1=vecphi1-rho*vecphi1_check
        matphi=matphi-rho*matphi_check
        matpsi=matpsi-rho*matpsi_check
             # for the calcul of gamma 
        gamma_vecphi0_tilde=vecphi0_tilde.copy()         #
        gamma_vecphi1_tilde=vecphi1_tilde.copy()
    
        vecphi0_tilde=vecphi0_tilde-rho*vecphi0_line
        vecphi1_tilde=vecphi1_tilde-rho*vecphi1_line
        
        #stopping criteria
        t1=np.sqrt(integ_deltaphi_deltapsi(h1,h2,N1,N2,vecphi0_tilde,vecphi0_tilde)+integ_phi_psi(h1,h2,N1,N2,vecphi1_tilde,vecphi1_tilde))
        print(testcondition)
        testcondition=t1/cond_t2
        
        
        
        
        # new descent direction
        ter1=integ_deltaphi_deltapsi(h1,h2,N1,N2,vecphi0_tilde,vecphi0_tilde)+integ_phi_psi(h1,h2,N1,N2,vecphi1_tilde,vecphi1_tilde)
        
        ter2=integ_deltaphi_deltapsi(h1,h2,N1,N2,gamma_vecphi0_tilde,gamma_vecphi0_tilde)+integ_phi_psi(h1,h2,N1,N2,gamma_vecphi1_tilde,gamma_vecphi1_tilde)
        gamma=ter1/ter2
    

        vecphi0_check=vecphi0_tilde+gamma*vecphi0_check
        vecphi1_check=vecphi1_tilde+gamma*vecphi1_check
        
         #  go to n+1
        
        itern=itern+1
        
    
        
    return vecphi0,vecphi1    
    
    
def control_function(vecphi0_hat,vecphi1_hat,x,y,k,h1,h2,M,N1,N2,C1,C2):
    # Solve the ph_Hat system 
    matphi_hat=Solver_Wave(M,N1,N2,k,C1,C2,np.zeros((M+2,N2)),np.zeros((M+2,N1)),np.zeros((M+2,N2)),np.zeros((M+2,N1)),vecphi0_hat[1:,1:],vecphi1_hat[1:,1:])

    y0j=y0(x[1:-1],y[1:-1])
    y1j=y1(x[1:-1],y[1:-1])
    # solve the control system
    normalphi_1=(-1./h1)*matphi_hat[:,0,:]
    normalphi_3=(-1./h1)*matphi_hat[:,-1,:]
    
    normalphi_2=(-1./h2)*matphi_hat[:,:,0]
    normalphi_4=(-1./h2)*matphi_hat[:,:,-1]
    maty_system_contr=Solver_Wave(M,N1,N2,k,C1,C2,normalphi_1,normalphi_2,normalphi_3,normalphi_4,y0j,y1j)
    
    return maty_system_contr
    
    
  
def get_norm(y_final,M,N1,N2,h1,h2,k):
    y_Mplus1=y_final[-1,:,:]
    y_M=y_final[M,:,:]
    y_pri_Mplus1=(1./k)*(y_Mplus1-y_M)
    
    norm_y_Mplus1=np.sqrt(integ_phi_psi(h1,h2,N1,N2,y_Mplus1,y_Mplus1))
    print(norm_y_Mplus1)
    # norm of y^' at T in H^-1
    uu=solve_Dirichlet(N1,N2,h1,h2,k,y_M,y_Mplus1,np.zeros((N1,N2)))
    uuu=addzerosVh(uu,N1,N2)
    norm_y_Mplus1_prim=np.sqrt(integ_deltaphi_deltapsi(h1,h2,N1,N2,uuu,uuu))
    print(norm_y_Mplus1_prim)    

'''  
def get_normcontrol(vecphi0_hat,vecphi1_hat,x,r,k,h,M):
    matphi_hat=solve1d_wave(vecphi0_hat[1:],vecphi1_hat[1:],np.zeros(M+2),r,k,M)
    normalphi_hat_1=(-1./h)*matphi_hat[-1,:]
    normvh=np.sqrt(appInte(normalphi_hat_1[:-1],normalphi_hat_1[:-1],k))
    print(normvh)
    return normalphi_hat_1  
    
    
'''   
    
    
    
    
    


