import math
import numpy as np
import matplotlib.pyplot as plt

def get_dispersion_relation(Hnn,Hnnpi,ks):
    Hrows, Hcols = Hnn.shape
    n_ks = ks.size
    bands = np.zeros((Hrows,n_ks),dtype = float)

    for n in range(n_ks):
        k = ks[n]
        Hk = Hnn + np.exp(1.0j*k)*Hnnpi.conj().T + np.exp(-1.0j*k)*Hnnpi
        eigs, vecs = np.linalg.eigh(Hk)
        bands[:,n] = eigs

    return bands

n_ks = 500
ks = np.linspace(-np.pi,np.pi,n_ks)

Hnn = np.array([[0.0, -np.sqrt(2), -1.0, 0.0],\
                [-np.sqrt(2),-2.0, 0.0, 0.0],\
                [-1.0, 0.0, 0.0,0.0],\
                [0.0, 0.0, 0.0, 2.0]], dtype = float)
Hnnpi = np.array([[0.0,-np.sqrt(2.0), 1.0, 0.0],\
                 [0.0, 0.0, 0.0, 0.0], \
                 [0.0, 0.0, 0.0, 0.0], \
                 [0.0, 0.0, 0.0, 0.0]], dtype = float)

bands = get_dispersion_relation(Hnn,Hnnpi,ks)        

fig = plt.figure()
plt.plot(ks,bands[0,:])
plt.plot(ks,bands[1,:])
plt.plot(ks,bands[2,:])
plt.plot(ks,bands[3,:])


Hnn = np.array([[0.0, -np.sqrt(2.0), -1.0, 0.0 ],\
                [-np.sqrt(2.0),0.0, -np.sqrt(2.0),0.0],\
                [-1.0, -np.sqrt(2),0.0, -np.sqrt(2)],\
                [0.0,0.0,-np.sqrt(2),0.0]], dtype = float)
Hnnpi = np.array([[0.0,0.0,-1.0,-np.sqrt(2.0)],\
                 [0.0, 0.0, 0.0, 0.0], \
                 [0.0, 0.0, 0.0, 0.0], \
                 [0.0, 0.0, 0.0, 0.0]], dtype = float)

bands = get_dispersion_relation(Hnn,Hnnpi,ks)        

fig = plt.figure()
plt.plot(ks,bands[0,:])
plt.plot(ks,bands[1,:])
plt.plot(ks,bands[2,:])
plt.plot(ks,bands[3,:])


Hnn = np.array([[0.0, -np.sqrt(2)],
                [-np.sqrt(2),0.0]], dtype = float)
Hnnpi = np.array([[-1.0, -np.sqrt(2)],\
                [0.0, 0.0]], dtype = float)

bands = get_dispersion_relation(Hnn,Hnnpi,ks)        

fig = plt.figure()
plt.plot(ks,bands[0,:])
plt.plot(ks,bands[1,:])



Hnn = np.array([[-np.sqrt(2), 0.0],
                [0.0,np.sqrt(2)]], dtype = float)
Hnnpi = np.array([[-0.5 - 1.0/np.sqrt(2), -0.5 - 1.0/np.sqrt(2)],\
                  [-0.5 + 1.0/np.sqrt(2), -0.5 + 1.0/np.sqrt(2)]], dtype = float)

bands = get_dispersion_relation(Hnn,Hnnpi,ks)        

fig = plt.figure()
plt.plot(ks,bands[0,:])
plt.plot(ks,bands[1,:])



Hnn = np.array([[0.0, 0.0],
                [0.0,0.0]], dtype = float)
tAA = 1.0
tAB = 1.0
Hnnpi = np.array([[tAA, tAB],\
                  [-tAA**2/tAB, -tAA]], dtype = complex)

bands = get_dispersion_relation(Hnn,Hnnpi,ks)        

fig = plt.figure()
plt.plot(ks,bands[0,:])
plt.plot(ks,bands[1,:])

plt.show()

