import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit
from numpy import linalg as LA
from scipy import linalg

eL = 0.0
eR = 1.0

DeltaL = 0.0
DeltaR = 0.0

tLQ = 1.0
tRQ = 0.1

Hsys = np.matrix([[eL-DeltaL,tLQ,0.0],[np.conj(tLQ),0.0,np.conj(tRQ)],[0.0,tRQ,eR-DeltaR]],dtype=complex)

n_ts = 400
ts = np.linspace(0,100,n_ts)

v0 = np.matrix([[1.0],[0.0],[0.0]],dtype=complex)

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
