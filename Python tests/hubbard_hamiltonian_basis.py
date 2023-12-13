import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit


np.set_printoptions(threshold=sys.maxsize,precision=3)

def fd(E,T):
    if np.abs(T) < 1e-5:
        if E > 0.0:
            return 0.0
        else:
            return 1.0

    return 1.0/(1.0+np.exp(E/T))

tL = 30
tR = 30

# assemble single particle Hamiltonian
#eA = -2.0
#eB = -2.0
#eA = 3.236
#eB = 3.236
#eA = 1.414
#eB = 1.414
eA = 0.0
eB = 0.0
tAA = 1.0
tAB = np.sqrt(2.0)*tAA
VB = -tAA
#VB = 0.0

U = -1.0
tSL = 1.0

sigmaz = np.matrix([[1.0,0.0],[0.0,-1.0]],dtype=complex)

Ncells = 5
Norbs = 2
on_site_energies = np.matrix([[eA,0],[0,eB]],dtype=complex)
intra_cell_hopping = np.matrix([[0,tAB],[np.conj(tAB),0]], dtype=complex)
inter_cell_hopping = np.matrix([[tAA,tAB],[0.0,0.0]], dtype=complex)
deleted_sites_m = []
deleted_sites_p = [1]

Nsites = Ncells*Norbs-len(deleted_sites_m)-len(deleted_sites_p)

H0 = np.zeros((Ncells*Norbs,Ncells*Norbs),dtype = complex)

I0 = np.eye(Nsites,dtype = complex)

for i in range(Ncells):
    H0[i*Norbs:(i+1)*Norbs,i*Norbs:(i+1)*Norbs] = on_site_energies +intra_cell_hopping
    if i > 0:
        H0[i*Norbs:(i+1)*Norbs,(i-1)*Norbs:(i)*Norbs] = inter_cell_hopping
    if i < Ncells-1:
        H0[i*Norbs:(i+1)*Norbs,(i+1)*Norbs:(i+2)*Norbs] = np.conj(inter_cell_hopping.T)

for i in deleted_sites_m:
    H0 = np.delete(H0,i,0)
    H0 = np.delete(H0,i,1)

for i in deleted_sites_p:
    H0 = np.delete(H0,-Norbs+i,0)
    H0 = np.delete(H0,-Norbs+i,1)

H0[0,0] = VB
H0[-1,-1] = VB

eigs,vecs = np.linalg.eigh(H0)
#for n in range(Ncells-2):
#    flat_band_state = np.zeros(Nsites,dtype=complex)
#    flat_band_state[1+2*n] =  -1.0/2
#    flat_band_state[1+2*n+1] = 1.0/np.sqrt(2)
#    flat_band_state[1+2*n+2] = -1.0/2
#    print("Flat band test")
#    print(np.matmul(H0,flat_band_state))
#    vecs[:,-Ncells+2+n] = flat_band_state

flat_band_vecs = vecs[:,-Ncells+2:]

# Generate matrix for Hubbard term in site basis, size Nsites**2 * Nsites**2
# The two-particle site basis is the Kronecker product of single particle site basis
Hub = np.zeros([Nsites**2,Nsites**2],dtype=complex)
for i in range(Nsites):
    Hub[i+i*Nsites,i+i*Nsites] = U

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.matshow(np.real(vecs),interpolation='nearest')
ax1.set_xticks(np.arange(0.5,len(vecs)-0.5,1))
ax1.set_yticks(np.arange(0.5,len(vecs)-0.5,1))
ax1.set_title("Eigenvectors of H0")
ax1.grid()

ax1 = fig.add_subplot(122)
ax1.matshow(np.imag(vecs),interpolation='nearest')
ax1.set_xticks(np.arange(0.5,len(vecs)-0.5,1))
ax1.set_yticks(np.arange(0.5,len(vecs)-0.5,1))
ax1.set_title("Eigenvectors of H0")
ax1.grid()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.matshow(np.real(Hub),interpolation='nearest')
ax1.set_xticks(np.arange(0.5,len(Hub)-0.5,1))
ax1.set_yticks(np.arange(0.5,len(Hub)-0.5,1))
ax1.set_title("Hubbard term in two-particle site basis")
ax1.grid()

for i in range(Nsites-1):
    ax1.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax1.axhline(y = (i+1)*Nsites-0.5,color='red')

site_to_sp = vecs.conj().T
#site_to_sp = np.linalg.inv(vecs)
sp_to_site = vecs


H0_sp = np.matmul(site_to_sp,np.matmul(H0,sp_to_site))

Hub_sp = np.matmul(np.matmul(np.kron(site_to_sp,site_to_sp),Hub),np.kron(sp_to_site,sp_to_site))


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.matshow(np.real(Hub_sp),interpolation='nearest')
ax2.set_xticks(np.arange(0.5,len(Hub)-0.5,1))
ax2.set_yticks(np.arange(0.5,len(Hub)-0.5,1))
ax2.set_title("Hubbard term in single particle eigenbasis")
ax2.grid()

for i in range(Nsites-1):
    ax2.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax2.axhline(y = (i+1)*Nsites-0.5,color='red')

Hub_sp_eigs, Hub_sp_vecs = np.linalg.eigh(Hub_sp);

print(Hub_sp_eigs)



fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.matshow(np.real(Hub_sp_vecs),interpolation='nearest')
ax2.set_xticks(np.arange(0.5,len(Hub_sp_vecs)-0.5,1))
ax2.set_yticks(np.arange(0.5,len(Hub_sp_vecs)-0.5,1))
ax2.set_title("Eigenvectors two-particle Hubbard term in single particle basis")
ax2.grid()

for i in range(Nsites-1):
    ax2.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax2.axhline(y = (i+1)*Nsites-0.5,color='red')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.matshow(np.real(np.diag(Hub_sp_eigs)),interpolation='nearest')
ax2.set_xticks(np.arange(0.5,len(Hub_sp_vecs)-0.5,1))
ax2.set_yticks(np.arange(0.5,len(Hub_sp_vecs)-0.5,1))
ax2.set_title("Eigenenergies of two-particle Hubbard term in single particle basis")
ax2.grid()

for i in range(Nsites-1):
    ax2.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax2.axhline(y = (i+1)*Nsites-0.5,color='red')

#plt.show()
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.matshow(np.imag(Hub_sp),interpolation='nearest')
#ax2.set_xticks(np.arange(0.5,len(Hub)-0.5,1))
#ax2.set_yticks(np.arange(0.5,len(Hub)-0.5,1))
#ax2.set_title("Imaginary Hubbard term in single particle eigenbasis")
#ax2.grid()

#for i in range(Nsites-1):
#    ax2.axvline(x = (i+1)*Nsites-0.5,color='red')
#    ax2.axhline(y = (i+1)*Nsites-0.5,color='red')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.matshow(np.real(H0_sp),interpolation='nearest')
ax3.set_xticks(np.arange(0.5,len(H0)-0.5,1))
ax3.set_yticks(np.arange(0.5,len(H0)-0.5,1))
ax3.set_title("H0 term in single particle eigenbasis")
ax3.grid()

H0_2p = np.kron(I0,H0) + np.kron(H0,I0)


H0_2p_sp = np.matmul(np.matmul(np.kron(site_to_sp,site_to_sp),H0_2p),np.kron(sp_to_site,sp_to_site))

eigs_H0_2p,vecs_H0_2p = np.linalg.eigh(H0_2p_sp)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.matshow(np.real(vecs_H0_2p),interpolation='nearest')
ax4.set_xticks(np.arange(0.5,len(vecs_H0_2p)-0.5,1))
ax4.set_yticks(np.arange(0.5,len(vecs_H0_2p)-0.5,1))
ax4.set_title("eigenvectors for H0 for two particles")
ax4.grid()

for i in range(Nsites-1):
    ax4.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax4.axhline(y = (i+1)*Nsites-0.5,color='red')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.matshow(np.diag(np.real(eigs_H0_2p)),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(eigs_H0_2p)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(eigs_H0_2p)-0.5,1))
ax5.set_title("eigevaluse for H0 in single particle basis")
ax5.grid()


fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.matshow(np.real(H0_2p),interpolation='nearest')
ax4.set_xticks(np.arange(0.5,len(H0_2p)-0.5,1))
ax4.set_yticks(np.arange(0.5,len(H0_2p)-0.5,1))
ax4.set_title("H0 for two particles in site basis")
ax4.grid()

for i in range(Nsites-1):
    ax4.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax4.axhline(y = (i+1)*Nsites-0.5,color='red')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.matshow(np.real(H0_2p_sp),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(H0_2p_sp)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(H0_2p_sp)-0.5,1))
ax5.set_title("H0 for two particles in single particle basis")
ax5.grid()

for i in range(Nsites-1):
    ax5.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax5.axhline(y = (i+1)*Nsites-0.5,color='red')

Hint_sp = H0_2p_sp + Hub_sp
Hint = H0_2p + Hub

eigs_tb_sp,vecs_tb_sp = np.linalg.eigh(Hint_sp)

eigs_tb,vecs_tb = np.linalg.eigh(Hint)

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.matshow(np.real(vecs_tb_sp),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(vecs_tb_sp)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(vecs_tb_sp)-0.5,1))
ax5.set_title("Eigenvectors two body Hamiltonian in single particle basis")
ax5.grid()

for i in range(Nsites-1):
    ax5.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax5.axhline(y = (i+1)*Nsites-0.5,color='red')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.matshow(np.real(np.diag(eigs_tb_sp)),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(vecs_tb_sp)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(vecs_tb_sp)-0.5,1))
ax5.set_title("Eigenenergies of two body Hamiltonian n single particle basis")
ax5.grid()

for i in range(Nsites-1):
    ax5.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax5.axhline(y = (i+1)*Nsites-0.5,color='red')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.matshow(np.real(vecs_tb),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(vecs_tb_sp)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(vecs_tb_sp)-0.5,1))
ax5.set_title("Eigenvectors two body Hamiltonian n single particle basis")
ax5.grid()

for i in range(Nsites-1):
    ax5.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax5.axhline(y = (i+1)*Nsites-0.5,color='red')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.matshow(np.real(np.diag(eigs_tb)),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(vecs_tb_sp)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(vecs_tb_sp)-0.5,1))
ax5.set_title("Eigenenergies of two body Hamiltonian n single particle basis")
ax5.grid()

for i in range(Nsites-1):
    ax5.axvline(x = (i+1)*Nsites-0.5,color='red')
    ax5.axhline(y = (i+1)*Nsites-0.5,color='red')

plt.show()

print(eigs_tb)

site_to_sp_2p = vecs_tb.conj().T
sp_to_site_2p = vecs_tb

HSL0 = np.zeros([Nsites,1],dtype=complex)
HSL0[0,0] = tSL
zero0 = np.zeros([Nsites,1],dtype=complex)
HSL_2p = np.kron(HSL0,zero0) + np.kron(zero0,HSL0)

print(HSL_2p)

HSL_2p_sp = np.matmul(sp_to_site_2p,HSL_2p)

print(HSL_2p_sp)
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.matshow(np.diag(np.real(HSL_2p)),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(HSL_2p)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(HSL_2p)-0.5,1))
ax5.set_title("System lead hopping, site basis real part two particles")
ax5.grid()

fig5 = plt.figure()
ax5 = fig5.add_subplot(121)
ax5.matshow(np.diag(real(HSL_2p_sp)),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(HSL_2p_sp)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(HSL_2p_sp)-0.5,1))
ax5.set_title("System lead hopping, sp basis real part two particles")
ax5.grid()

ax5 = fig5.add_subplot(122)
ax5.matshow(np.diag(np.imag(HSL_2p_sp)),interpolation='nearest')
ax5.set_xticks(np.arange(0.5,len(HSL_2p_sp)-0.5,1))
ax5.set_yticks(np.arange(0.5,len(HSL_2p_sp)-0.5,1))
ax5.set_title("System lead hopping, sp basis imag part two particles")
ax5.grid()




# Assemble Hamiltonian with an extra site
H0_extra_site = np.block([[0.0, HSL0.conj().T],[HSL0,H0]],dtype = complex)
H0_2p_extra_site = np.kron(np.eye(Nsites+1),H0_extra_site))
H0_2p_extra_site += np.kron(H0_extra_site,np.eye(Nsites+1))

# shuffle the lead related indices first

#Hhub_extra_site = np.zeros([




