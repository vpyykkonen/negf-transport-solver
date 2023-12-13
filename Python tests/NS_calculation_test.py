import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit
import scipy.linalg as la

def fd(E,T):
    if np.abs(T) < 1e-5:
        if E > 0.0:
            return 0.0
        else:
            return 1.0

    return 1.0/(1.0+np.exp(E/T))

def prepare_H0(Norbs, Ncells, on_site, intra_cell, inter_cell,\
        deleted_site_m, deleted_sites_p):
    Nsites = Ncells*Norbs-len(deleted_sites_m)-len(deleted_sites_p)
    H0 = np.zeros((Ncells*Norbs,Ncells*Norbs),dtype = complex)
    for i in range(Ncells):
        H0[i*Norbs:(i+1)*Norbs,i*Norbs:(i+1)*Norbs] = on_site + intra_cell
        if i > 0:
            H0[i*Norbs:(i+1)*Norbs,(i-1)*Norbs:(i)*Norbs] = inter_cell
        if i < Ncells-1:
            H0[i*Norbs:(i+1)*Norbs,(i+1)*Norbs:(i+2)*Norbs] = np.conj(inter_cell.T)
    
    for i in deleted_sites_m:
        H0 = np.delete(H0,i,0)
        H0 = np.delete(H0,i,1)
    
    for i in deleted_sites_p:
        H0 = np.delete(H0,-Norbs+i,0)
        H0 = np.delete(H0,-Norbs+i,1)
    return Nsites, H0

def get_current(HBdG,tLS,tRS,muL,DeltaR,tL,tR,TL,TR,ieta):
    n_Es = 500
    Es = np.linspace(0.0,muL,n_Es)
    
    cur = 0
    
    HBdG_rows,cols = HBdG.shape
    
    Sigma = np.zeros((HBdG_rows+4,HBdG_rows+4),dtype = complex)
    Sigma[0,2] = tLS
    Sigma[1,3] = -np.conj(tLS)
    Sigma[2,0] = np.conj(tLS)
    Sigma[3,1] = -tLS
    Sigma[-2,-4] = tRS
    Sigma[-1,-3] = -np.conj(tRS)
    Sigma[-4,-2] = np.conj(tRS)
    Sigma[-3,-1] = -tRS
    
    for E in Es:
        # Left lead g
        phi_p = np.arccos((E+ieta)/(2.0*tL))
        phi_h = np.arccos((E+ieta)/(2.0*tL))
        gRLL = np.array([[np.exp(-1.0j*phi_p)/tL,0.0],\
                        [0.0, np.exp(-1.0j*phi_h)/tL]], dtype = complex)
        glLL = fd(E-muL,TL)*(gRLL.conj().T-gRLL)
        # right lead g
        gRRR = np.array([[-(E+ieta), DeltaR],\
                        [np.conj(DeltaR), -(E+ieta)]], dtype = complex)
        gRRR *= 1.0/(tR*np.sqrt(np.abs(DeltaR)**2-(E+ieta)**2))
        glRR = fd(E-muR,TR)*(gRRR.conj().T-gRRR)

    
        # system g
        gRSS = np.linalg.inv((E+ieta)*np.eye(HBdG_rows)-HBdG)
    
        gR = la.block_diag(gRLL,gRSS,gRRR)
        gl = la.block_diag(glLL,np.zeros(gRSS.shape),glRR)
    
    
        GR = np.linalg.solve(np.eye(HBdG_rows+4)-np.matmul(gR,Sigma),gR)
        Gl = np.matmul(np.matmul(np.eye(HBdG_rows+4) + np.matmul(GR,Sigma),gl),\
                np.eye(HBdG_rows+4) + np.matmul(Sigma,GR.conj().T))
    
        #cur += -4.0*np.real(tLS*Gl[0,2])
        cur += np.abs(GR[2,3])**2*(fd(E+muL,TL)-fd(E-muL,TL))
    
    cur *= np.abs(tLS)**2/(np.pi*tL)*(muL-muR)/n_Es
    return cur

tL = 30
tR = 30

eA = 0.0
eB = 0.0
tAA = -1.0
tAB = -np.sqrt(2.0)*tAA
VB = 0.0

U = -0.0
tLS = 1.0
tRS = 1.0


sigmaz = np.matrix([[1.0,0.0],[0.0,-1.0]],dtype=complex)

Ncells = 1
Norbs = 2
on_site_energies = np.matrix([[eA,0],[0,eB]],dtype=complex)
intra_cell_hopping = np.matrix([[0,tAB],[np.conj(tAB),0]], dtype=complex)
inter_cell_hopping = np.matrix([[tAA,tAB],[0.0,0.0]], dtype=complex)
deleted_sites_m = []
deleted_sites_p = [1]

Nsites, H0 = prepare_H0(Norbs, Ncells, on_site_energies, intra_cell_hopping,\
        inter_cell_hopping, deleted_sites_m, deleted_sites_p)

#H0[0,0] = VB
#H0[-1,-1] = VB
print(H0)

muL = 0.1
muR = 0.0
TL = 0.0
TR = 0.0
DeltaL = 0.0
DeltaR = 1.0
ieta = 1.0e-4j


n_gates = 100
gates = np.linspace(-0.5,0.5,n_gates)
curs = np.zeros(n_gates)
for n in range(n_gates):
    gate = gates[n]
    gate_m = np.zeros(H0.shape)
    np.fill_diagonal(gate_m,gate)
    HBdG = np.kron(H0-gate_m,sigmaz) 
    curs[n] = get_current(HBdG,tLS,tRS,muL,DeltaR,tL,tR,TL,TR,ieta)
    print("Gate: ", gate, " current: ", curs[n])

fig = plt.figure()
plt.plot(gates,curs)
plt.show()









    







