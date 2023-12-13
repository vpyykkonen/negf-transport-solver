import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit
from numpy import linalg as LA
from scipy import linalg
import matplotlib.animation as animation

#import networkx as nx
#from networkx import *

#G = nx.Graph()
#G.add_nodes_from(["L","QD","R"])
#G.add_edges_from([("L","QD"),("QD","R")])
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#nx.draw(G,pos=nx.bipartite_layout(G,G.nodes),with_labels=True)
#plt.show()
#
#exit()


np.set_printoptions(threshold=sys.maxsize,precision=3)

def fd(E,T):
    if np.abs(T) < 1e-5:
        if E > 0.0:
            return 0.0
        else:
            return 1.0

    return 1.0/(1.0+np.exp(E/T))

eL = 0.0
eR = 0.3

n_eRs = 9
eRs = np.linspace(0.0,0.8,n_eRs)

fig = plt.figure()
ax = plt.axes()
DeltaL = 0.0
DeltaR = 0.0

tLQ = 1.0
tRQ = 1.0

# initial vector
phase = 0
L0_amp = 1/np.sqrt(2)

n_phases = 10
phases = np.linspace(0,np.pi,n_phases)

v0 = np.matrix([[1/np.sqrt(2)],[0.0],[np.sqrt(1-np.abs(L0_amp)**2)*np.exp(1.0j*phase)]],dtype=complex)
textstr = '\n'.join((
    r'$|\psi(t=0)\rangle = %.2f|L\rangle + %.2f \exp(i $phase$ )|R\rangle$' % (np.abs(v0[0]),np.abs(v0[2])),
    r'$\epsilon_{L} = %.2f$' % (eL),
    r'$\epsilon_{R} = %.2f$' % (eR),
    r'$t_{L,QD} = %.2f$' % (tLQ),
    r'$t_{R,QD} = %.2f$' % (tRQ),
    r'$t_{L,R} = 0$'))
ax.text(0.05,0.95,textstr,transform=ax.transAxes,
        fontsize=12,verticalalignment='top')

#for n_eR in range(n_eRs)

for j in range(n_phases):

    #eR = eRs[j]
    phase = phases[j]
    n_eQs = 300
    eQs = np.linspace(-20.0,20.0,n_eQs)
    v0 = np.matrix([[1/np.sqrt(2)],[0.0],[np.sqrt(1-np.abs(L0_amp)**2)*np.exp(1.0j*phase)]],dtype=complex)




    HL_BdG = np.matrix([[eL,DeltaL],[np.conj(DeltaL),-eL]], dtype=complex)
    eigs_L,vecs_L = LA.eigh(HL_BdG)

    tLQ_BdG = np.matrix([[tLQ,0.0],[0.0,-np.conj(tLQ)]],dtype=complex)

    tLQ_diag = np.matmul(vecs_L.conj().T,tLQ_BdG)

    print(eigs_L)
    print(tLQ_diag)

    HR_BdG = np.matrix([[eR,DeltaR],[np.conj(DeltaR),-eR]], dtype=complex)
    eigs_R,vecs_R = LA.eigh(HR_BdG)

    tRQ_BdG = np.matrix([[tRQ,0.0],[0.0,-np.conj(tRQ)]],dtype=complex)

    tRQ_diag = np.matmul(vecs_R.conj().T,tRQ_BdG)
    print(eigs_R)
    print(tRQ_diag)

    # Hamiltonian of the three-level system
    Hsys = np.matrix([[eL-DeltaL,tLQ,0.0],[np.conj(tLQ),0.0,np.conj(tRQ)],[0.0,tRQ,eR-DeltaR]],dtype=complex)
    # Hamiltonian parameters from the BdG lead
    #Hsys = np.matrix([[eigs_L[0],tLQ_diag[0,0],0.0],[np.conj(tLQ_diag[0,0]),0.0,np.conj(tRQ_diag[0,0])],[0.0,tRQ_diag[0,0],eigs_R[0]]],dtype=complex)
    #

    print("Hsys",Hsys)
    eigs,vecs = LA.eigh(Hsys)
    print("eigs",eigs)
    print("vecs",vecs)


    # Current operator from left lead to system
    I_QLop = np.matrix([[0.0,1.0j*np.conj(tLQ),0.0],[-1.0j*tLQ,0.0,0.0],[0.0,0.0,0.0]],dtype=complex)

    n_ts = 200
    ts = np.linspace(0,300,n_ts)



    I_vars = np.zeros(n_eQs)
    I_RMSs = np.zeros(n_eQs)
    I_flucts = np.zeros(n_eQs)

    count1 = 0
    for eQ in eQs:
        Hsys[1,1] = eQ
        eigs,vecs = LA.eigh(Hsys)

        I_expts = np.zeros(n_ts)

        # initial vector in eigenbasis
        v0_eig = np.matmul(vecs.conj().T,v0)

        # matrix of I in eigenbasis
        I_QL_eig = np.matmul(vecs.conj().T,np.matmul(I_QLop,vecs))
        I_QL_2_eig = np.matmul(I_QL_eig,I_QL_eig)

        I_RMS = 0.0
        for n in range(0,3):
            for m in range(n+1,3):
                I_RMS += 2.0*np.abs(np.square(v0_eig[n]*I_QL_eig[n,m]*v0_eig[m]))

        I_RMSs[count1] = np.sqrt(I_RMS)
        I_fluct = 0.0
        for n in range(0,3):
            I_fluct += (np.abs(v0_eig[n])**2*I_QL_2_eig[n,n])[0,0]

        I_flucts[count1] = np.sqrt(I_fluct)

        count2 = 0
        for t in ts:
            U = np.matmul(vecs,np.matmul(np.diag(np.exp(-1.0j*t*eigs)),vecs.conj().T))
            #U = linalg.expm(-1.0j*t*Hsys)

            IQLop_t = np.matmul(U.conj().T,np.matmul(I_QLop,U))
            I_expt = np.matmul(v0.conj().T,np.matmul(IQLop_t,v0))[0,0]
            I_expts[count2] = I_expt
            count2 += 1
        I_vars[count1] = np.sqrt(np.mean(np.square(I_expts)))
        count1 += 1


    #fig = plt.figure()
    #ax = fig.add_subplot(131)
    #ax.plot(eQs,I_vars)

    #ax = fig.add_subplot(111)
    ax.plot(eQs,I_RMSs,label='phase$_R = %.2f$'%(phase))


    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(eQs,I_flucts)

    #plt.show()


ax.set_xlabel("$\epsilon_{QD}$",fontsize=12)
ax.set_ylabel("RMS $I_{L,QD}$ expectation value",fontsize=12)
ax.legend()
plt.show()




