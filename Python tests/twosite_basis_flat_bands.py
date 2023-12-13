import numpy as np
from matplotlib import pyplot as plt

a = 1.0
epsA = 0.0
epsB = -5.0
tAA_R = 1.0
tAA_L = np.conj(tAA_R)
tBB_R = 0.0
tBB_L = np.conj(tBB_R)
tAB_R = 0.0
tBA_L = np.conj(tAB_R)
tBA_R = np.sqrt(2*tAA_R**2+(epsB-epsA)*tAA_R)
tAB_L = np.conj(tBA_R)
tAB_0 = np.sqrt(2*tAA_R**2+(epsB-epsA)*tAA_R)
tBA_0 = np.conj(tAB_0)
#tAB_0 = 2.8
#tBA_0 = np.conj(tAB_0)

print("E = ",(tBA_R*tAB_0+tAB_R*tBA_0)/(tAA_R+tBB_R))

def ham(k):
    ham = np.zeros([2,2],dtype=complex)
    ham[0,0] = -tAA_R*np.exp(1.0j*k*a) - tAA_L*np.exp(-1.0j*k*a)
    ham[0,1] = -tAB_R*np.exp(1.0j*k*a) - tAB_L*np.exp(-1.0j*k*a) - tAB_0
    ham[1,0] = -tBA_R*np.exp(1.0j*k*a) - tBA_L*np.exp(-1.0j*k*a) - tBA_0
    ham[1,1] = -tBB_R*np.exp(1.0j*k*a) - tBB_L*np.exp(-1.0j*k*a)
    return ham

n_ks = 1001
ks = np.linspace(-np.pi,np.pi,n_ks)

Es = np.zeros([n_ks,2])

for n in range(n_ks):
    ham_k = ham(ks[n])
    eigs,vecs = np.linalg.eigh(ham_k)
    Es[n,:] = eigs

fig,ax = plt.subplots()
ax.plot(ks,Es[:,0])
ax.plot(ks,Es[:,1])
plt.show()



