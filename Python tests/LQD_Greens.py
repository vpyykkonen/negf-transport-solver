import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit
from numpy import linalg as LA
from scipy import linalg

def fd(E,T):
    if np.abs(T) < 1e-5:
        if E > 0.0:
            return 0.0
        else:
            return 1.0

    return 1.0/(1.0+np.exp(E/T))

# System parameters

eQD = 0.0 # quantum dot energy

tL = 30.0 # lead hopping
DeltaL = 1.0 # lead superconducting order parameter

tLQD = 5.3 # QD lead hopping

ieta = 1.0e-3j # regularization

TL = 0.0 # lead temperature

n_Es = 1000
Es = np.linspace(-5,5,n_Es)

# Nambu LQD hopping matrix
HLQD = np.matrix([[tLQD,0.0],[0.0,-tLQD]],dtype=complex)
# Nambu QD Hamiltonian
HQD = np.matrix([[eQD,0.0],[0.0,-eQD]],dtype=complex)

L_pdos = np.zeros(n_Es)
L_hdos = np.zeros(n_Es)
QD_pdos = np.zeros(n_Es)
QD_hdos = np.zeros(n_Es)

for n in range(n_Es):
    # Lead Green
    E = Es[n]
    gRLL_coeff = 1.0/(tL*np.sqrt(np.abs(DeltaL)**2-(E+ieta)**2))
    gRLL = gRLL_coeff*np.matrix([[-(E+ieta),DeltaL],[np.conj(DeltaL),-(E+ieta)]],dtype=complex)

    GRSS = np.linalg.inv((E+ieta)*np.eye(2)-HQD-np.matmul(HLQD.conj().T,np.matmul(gRLL,HLQD)))

    L_pdos[n] = -np.imag(gRLL[0,0])
    L_hdos[n] = -np.imag(gRLL[1,1])
    QD_pdos[n] = -np.imag(GRSS[0,0])
    QD_hdos[n] = -np.imag(GRSS[1,1])

print('sum QD_pdos',(Es[2]-Es[1])*np.sum(QD_pdos))
print('sum QD_pdos',(Es[2]-Es[1])*np.sum(QD_hdos))

fig1 = plt.figure()
ax11 = fig1.add_subplot(121)
ax11.plot(Es,L_pdos)

ax12 = fig1.add_subplot(122)
ax12.plot(Es,L_hdos)

fig2 = plt.figure()
ax21 = fig2.add_subplot(121)
ax21.plot(Es,QD_pdos)

ax22 = fig2.add_subplot(122)
ax22.plot(Es,QD_hdos)


plt.show()
