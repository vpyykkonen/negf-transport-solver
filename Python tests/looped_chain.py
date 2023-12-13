import numpy as np

from numpy import linalg as LA

from matplotlib import pyplot as plt
from plot_matrix import plot_matrix

L = 12
t = 1.0+0.0j

H0 = np.zeros((L+1,L+1),dtype=complex)

H0[0,L-1] = t
H0[L-1,0] = np.conj(t)

for i in range(L-1):
    H0[i,i+1] = t
    H0[i+1,i] = np.conj(t)
H0[0,L] = t
H0[L,0] = np.conj(t)
H0[L,int(L/2)] = np.conj(t)
H0[int(L/2),L] = t
H0[2,2] = 1.0


Es,V = LA.eigh(H0)
print(Es)
plot_matrix(H0)
plot_matrix(V)
plt.show()


