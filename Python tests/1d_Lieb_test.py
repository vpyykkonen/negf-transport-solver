import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

t = 1.0
a = 1.0

def H0_k(k):
    H0k = np.zeros((5,5),dtype=complex)
    H0k[0,1] = t*(1.0+np.exp(1.0j*k*a))
    H0k[1,0] = t*(1.0+np.exp(-1.0j*k*a))
    H0k[2,0] = t
    H0k[0,2] = t
    H0k[3,2] = t
    H0k[2,3] = t
    H0k[4,3] = t*(1.0+np.exp(1.0j*k*a))
    H0k[3,4] = t*(1.0+np.exp(-1.0j*k*a))
    return H0k

n_ks = 100
ks = np.linspace(-np.pi/a,np.pi/a,n_ks)
Es_k = np.zeros((5,n_ks))
for i in range(n_ks):
    Es,V = LA.eigh(H0_k(ks[i]))
    Es_k[:,i] = Es

fig = plt.figure()
plt.plot(ks,Es_k[0,:])
plt.plot(ks,Es_k[1,:])
plt.plot(ks,Es_k[2,:])
plt.plot(ks,Es_k[3,:])
plt.plot(ks,Es_k[4,:])
plt.show()
    


