import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

Delta = 1.0
t = 1.0
T = 0.1

n_thetas = 100
thetas = np.linspace(0,2*np.pi,n_thetas)

def fd_dist(E,T):
    if T < 1e-5:
        if E > 0:
            return 0.0
        else:
            return 1.0
    else:
        return 1.0/(np.exp(E/T)+1.0)
        

Is = np.zeros([4,n_thetas])
I_tot = np.zeros([n_thetas])
Is_alt = np.zeros([4,n_thetas])
I_tot_alt = np.zeros([n_thetas])
for num in range(n_thetas):
    theta = thetas[num]
    HBdG = np.matrix([[0.0,Delta,t,0.0],[Delta,0.0,0.0,-t],[t,0.0,0.0,Delta*np.exp(1.0j*theta)],[0.0,-t,Delta*np.exp(-1.0j*theta),0.0]],dtype=complex)
    eigs,vecs = LA.eigh(HBdG)
    Is_alt[0,num] = -Delta*t*np.cos(theta/2)/eigs[0]
    Is_alt[1,num] = Delta*t*np.cos(theta/2)/eigs[1]
    Is_alt[2,num] = Delta*t*np.cos(theta/2)/eigs[2]
    Is_alt[3,num] = -Delta*t*np.cos(theta/2)/eigs[3]
    for n in range(4):
        Is[n,num] = 4*np.imag(t*np.conj(vecs[0,n])*vecs[2,n])
        I_tot[num] += Is[n,num]*fd_dist(eigs[n],T)
        I_tot_alt[num] += Is_alt[n,num]*fd_dist(eigs[n],T)


fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(thetas, Is[0,:])
ax = fig.add_subplot(222)
ax.plot(thetas, Is[1,:])
ax = fig.add_subplot(223)
ax.plot(thetas, Is[2,:])
ax = fig.add_subplot(224)
ax.plot(thetas, Is[3,:])

fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(thetas, Is_alt[0,:])
ax = fig.add_subplot(222)
ax.plot(thetas, Is_alt[1,:])
ax = fig.add_subplot(223)
ax.plot(thetas, Is_alt[2,:])
ax = fig.add_subplot(224)
ax.plot(thetas, Is_alt[3,:])


fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(thetas, Is[0,:]+Is[1,:])
ax.plot(thetas, I_tot)

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(thetas, Is[0,:]+Is[1,:])
ax.plot(thetas, I_tot_alt)

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(thetas, Is[0,:]+Is[1,:])
ax.plot(thetas, np.abs(I_tot - I_tot_alt))

plt.show()

