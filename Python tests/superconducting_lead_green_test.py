import numpy as np
from matplotlib import pyplot as plt

def lead_green(E,tL,DeltaL,ieta):
    gRLL11 = (E/(2*tL**2))*(1-np.sqrt((1+4*tL**2/(np.abs(DeltaL)**2-E**2)).astype(complex)))
    gRLL12 = DeltaL*gRLL11/E
    gRLL21 = np.conj(DeltaL)*gRLL11/E
    return np.array([[gRLL11,gRLL12],[gRLL21,gRLL11]])

DeltaL = 1.0+0.0j
tL = 10
ieta = 1e-3j


n_Es = 5000
Es = np.linspace(-5*tL,5*tL,n_Es)
gRLLs = np.zeros([n_Es,2,2],dtype=complex)
for n in range(n_Es):
    gRLLs[n,:,:] = lead_green(Es[n],tL,DeltaL,ieta)

print(np.trapz(-np.imag(gRLLs[:,0,0])/(np.pi),Es))

fig,ax = plt.subplots()
#ax.plot(Es,-np.real(gRLLs[:,0,0]))
ax.plot(Es,-np.imag(gRLLs[:,0,0]))


plt.show()
