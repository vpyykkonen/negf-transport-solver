import numpy as np
from matplotlib import pyplot as plt



# hopping matrix for the sawtooth lattice
def hop_sawtooth(n_cells,tAA,tAB):
    n_sites = 2*n_cells + 1
    hop = np.zeros([n_sites,n_sites],dtype=complex)
    for n in range(n_cells):
        hop[2*(n+1),2*n] = tAA
        hop[2*n,2*(n+1)] = np.conj(tAA)
        hop[2*n+1,2*n] = tAB
        hop[2*n,2*n+1] = np.conj(tAB)
        hop[2*(n+1),2*n+1] = tAB
        hop[2*n+1,2*(n+1)] = np.conj(tAB)
    return hop

#def fd_dist(E,mu,T):
#    if np.abs(T)<1e-10:
#        if E < mu:
#            return 1.0
#        else:
#            return 0.0
#    else:
#        return 1.0/(1.0+np.exp((E-mu)/T))
        

n_cells = 3
tAA = -1.0
tAB = np.sqrt(2.0)*tAA
gate = -10.0

VCL = -tAA
VCR = -tAA

H_sawtooth = hop_sawtooth(n_cells,tAA,tAB)
H_sawtooth[0,0] = VCL
H_sawtooth[-1,-1] = VCR
n_sites = 2*n_cells+1

H_sawtooth += gate*np.eye(n_sites,dtype=complex)

print(H_sawtooth)


sigmaz = np.asarray([[1.0,0.0],[0.0,-1.0]],dtype=complex)


T = 0.01
tLS = -1.0
tRS = -1.0

HBdG = np.kron(H_sawtooth,sigmaz)

ieta = 1.0e-3j
tL = -30.0
DeltaL = 1.0

eState = 0.0



n_Es = 10000
Es = np.linspace(-40,3,n_Es)
SigmaR_Es = np.zeros([n_Es,2*n_sites,2*n_sites],dtype=complex)
gRSigmaR_Es = np.zeros([n_Es,2*n_sites,2*n_sites],dtype=complex)
GRSS_Es = np.zeros([n_Es,2*n_sites,2*n_sites],dtype=complex)

I_Nambu = np.eye(2*n_sites,dtype=complex)

for n in range(n_Es):
    E = Es[n]
    gRLL = np.zeros([2,2],dtype=complex)
    gRLL[0,0] = -(E+ieta)
    gRLL[1,1] = -(E+ieta)
    gRLL[0,1] = -DeltaL
    gRLL[1,0] = -np.conj(DeltaL)
    gRLL *= 1.0/(tL*np.sqrt(np.abs(DeltaL)**2-(E+ieta)**2))

    gRSS = np.linalg.inv((E+ieta)*I_Nambu-HBdG)

    SigmaR = np.zeros([2*n_sites,2*n_sites],dtype=complex)
    SigmaR[0:2,0:2] = np.abs(tLS)**2*np.matmul(sigmaz,np.matmul(gRLL,sigmaz))
    GRSS = np.linalg.solve((E+ieta)*I_Nambu-HBdG-SigmaR,I_Nambu)
    GRSS_Es[n,:,:] = GRSS
    SigmaR_Es[n,:,:] = SigmaR


    gRSigmaR_Es[n,:,:] = np.matmul(gRSS,SigmaR)
    #GRSS_Es[n,:,:] = np.linalg.solve(np.eye(2,dtype=complex)-gRSigmaR_Es[n,:,:],gRSS)

fig,ax = plt.subplots()
ax.plot(Es,np.abs(SigmaR_Es[:,0,0]),label="00")
ax.plot(Es,np.abs(SigmaR_Es[:,0,1]),label="01")
ax.set_xlabel("E")
ax.set_ylabel("SigmaRLL")
ax.legend()

fig,ax = plt.subplots()
ax.semilogy(Es,np.abs(gRSigmaR_Es[:,0,0]),label="00")
ax.semilogy(Es,np.abs(gRSigmaR_Es[:,0,1]),label="01")
ax.set_xlabel("E")
ax.set_ylabel("gRSigmaRLL")
ax.legend()

fig,ax = plt.subplots()
ax.semilogy(Es,-np.imag(GRSS_Es[:,0,0]),label="00")
ax.semilogy(Es,-np.imag(GRSS_Es[:,0,1]),label="01")
ax.set_xlabel("E")
ax.set_ylabel("GRSS")
ax.legend()

fig,ax = plt.subplots()
ax.semilogy(Es,np.imag(GRSS_Es[:,0,0]),label="00")
ax.semilogy(Es,np.imag(GRSS_Es[:,0,1]),label="01")
ax.set_xlabel("E")
ax.set_ylabel("GRSS")
ax.legend()

print(np.trapz(2.0*np.imag(GRSS_Es[:,0,0]*1.0/(1.0+np.exp(Es/T))),Es)/(2.0*np.pi))
print(np.trapz(np.imag((GRSS_Es[:,0,1]-np.conj(GRSS_Es[:,1,0]))*1.0/(1.0+np.exp(Es/T))),Es)/(2.0*np.pi))

plt.show()




