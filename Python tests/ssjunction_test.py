import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.linalg import block_diag


def fd_dist(E,mu,T):
    if T < 1e-6:
        if E > mu:
            return 0.0
        else:
            return 1.0
    else:
        return 1.0/(1.0+np.exp((E-mu)/T))



def gRLL_and_glLL(E,muL,DeltaL,tL,TL,ieta):
    if np.abs(DeltaL) < 1e-6:
        phi = np.arccos((E+muL)/(2.0*tL))
        gR = np.exp(-1.0j*phi)/tL
        gl = -fd_dist(E,muL,tL)*np.imag(gR)
        return gR,gl
    else: 
        gR = np.zeros([2,2],dtype=complex)
        gl = np.zeros([2,2],dtype=complex)
        gR[0,0] = -(E+ieta)
        gR[1,1] = -(E+ieta)
        gR[0,1] = -DeltaL
        gR[1,0] = -np.conj(DeltaL)
        gR *= 1.0/(tL*np.sqrt((np.abs(DeltaL)**2-(E+ieta)**2).astype(complex)))
        gl = fd_dist(E,0.0,TL)*(np.conj(np.transpose(gR))-gR)
        return gR,gl

def gR_from_eigen(E,ham_eigs,ham_vecs,ieta):
    gReigs = 1.0/(E+ieta-ham_eigs)
    gReigs_vecs = np.multiply(gReigs[:,None],np.conj(np.transpose(ham_vecs)))
    gRSS = np.matmul(ham_vecs,gReigs_vecs)
    return gRSS

def assemble_SigmaR(sigmas,n_freqs):
    n_block = sigmas[0][0].shape[0]
    SigmaR = np.zeros([n_freqs*n_block,n_freqs*n_block],dtype=complex)
    for sigma,harm in sigmas:
        for n in range(max(0,-harm),min(n_freqs,n_freqs-harm)):
            SigmaR[(n+harm)*n_block:(n+harm+1)*n_block,n*n_block:(n+1)*n_block] = sigma
    return SigmaR

def Greens_freq_nn(E,ham_eigs,ham_vecs,SigmaR,base_freq,cutoff_below,cutoff_above,muL,DeltaL,tL,TL, DeltaR,tR,TR,ieta):
    n_freqs = math.ceil((cutoff_above-cutoff_below)/base_freq)
    n_block = ham_eigs.shape[0]
    nambu = np.abs(DeltaL) > 1e-6 or np.abs(DeltaR) > 1e-6
    if nambu:
        n_block += 4
    else:
        n_block += 2

    # Assemble Green's function matrices
    gR = np.zeros([n_freqs*n_block,n_freqs*n_block],dtype=complex)
    gl = np.zeros([n_freqs*n_block,n_freqs*n_block],dtype=complex)
    for n in range(n_freqs):
        En = cutoff_below + n*base_freq + E
        states_per_site = 1
        if nambu:
            states_per_site = 2
        # Get lead Green's functions
        gRLL = np.zeros([states_per_site,states_per_site],dtype=complex)
        glLL = np.zeros([states_per_site,states_per_site],dtype=complex)
        gRRR = np.zeros([states_per_site,states_per_site],dtype=complex)
        glRR = np.zeros([states_per_site,states_per_site],dtype=complex)

        if nambu: 
            if DeltaL < 1.0e-6: # Handle normal leads appropriately
                gRLLp,glLLp = gRLL_and_glLL(En,muL,0.0,tL,TL,ieta)
                gRLLh,glLLh = gRLL_and_glLL(En,-muL,0.0,tL,TL,ieta)
                gRLL = np.array([[gRLLp,0.0],[0.0,gRLLh]],dtype=complex)
                glLL = np.array([[glLLp,0.0],[0.0,glLLh]],dtype=complex)
            else:
                gRLL,glLL = gRLL_and_glLL(En,0.0,DeltaL,tL,TL,ieta)
            if DeltaR < 1.0e-6:
                gRRRp,glRRp = gRLL_and_glLL(En,muR,0.0,tR,TR,ieta)
                gRRRh,glRRh = gRLL_and_glLL(En,-muR,0.0,tR,TR,ieta)
                gRRR = np.array([[gRRRp,0.0],[0.0,gRRRh]],dtype=complex)
                glRR = np.array([[glRRp,0.0],[0.0,glRRh]],dtype=complex)
            else:
                gRRR,glRR = gRLL_and_glLL(En,0.0,DeltaR,tR,TR,ieta)
        else:
            gRLL,glLL = gRLL_and_glLL(En,muL,0.0,tL,TL,ieta)
            gRRR,glRR = gRLL_and_glLL(En,muR,0.0,tR,TR,ieta)

        gRSS = gR_from_eigen(En,ham_eigs,ham_vecs,ieta) # middle part Green's functions
        # Assemble non-perturbed Greens
        gRnn = block_diag(gRLL,gRSS,gRRR)
        glnn = block_diag(glLL,np.zeros([n_block-2*states_per_site,n_block-2*states_per_site],dtype=complex),glRR)
        gR[n*n_block:(n+1)*n_block,n*n_block:(n+1)*n_block] = gRnn
        gl[n*n_block:(n+1)*n_block,n*n_block:(n+1)*n_block] = glnn

    # Solve the perturbed Green's functions
    GR = np.linalg.solve(np.eye(n_freqs*n_block,dtype=complex)-np.matmul(gR,SigmaR),gR)
    IpGRSigmaR = np.eye(n_freqs*n_block)+np.matmul(GR,SigmaR)
    Gl = np.matmul(np.matmul(IpGRSigmaR,gl),np.conj(np.transpose(IpGRSigmaR)))
    return GR,Gl

def get_Gln(Gl,n,n_freqs,n_states):
    Gl_reshape = Gl.reshape([n_freqs,n_states,n_freqs,n_states])
    return np.trace(Gl_reshape,offset=n,axis1=0,axis2=2,dtype=complex)

# Define Hamiltonian
tAA = -1.0
#tAB = 0.9*np.sqrt(2.0)*tAA
tAB = -1.35
VB = -tAA
n_unitcells = 2
n_sites = 2*n_unitcells+1
ham = np.zeros([n_sites,n_sites],dtype=complex)
for n in range(n_unitcells):
    ham[2*n,2*n+1] = tAB
    ham[2*n+1,2*n] = tAB
    ham[2*n,2*n+2] = tAA
    ham[2*n+2,2*n] = tAA
    ham[2*n+1,2*n+2] = tAB
    ham[2*n+2,2*n+1] = tAB
mu = -2.5
np.fill_diagonal(ham, -mu)
ham[0,0] += VB
ham[-1,-1] += VB

sigmaz = np.array([[1.0,0.0],[0.0,-1.0]],dtype=complex)

ham_bdg = np.kron(ham,sigmaz)

fig,ax = plt.subplots()

pos = ax.matshow(np.real(ham_bdg),interpolation='nearest')
ax.set_xticks(np.arange(0.5,len(ham_bdg)-0.5,1))
ax.set_yticks(np.arange(0.5,len(ham_bdg)-0.5,1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title("Phase angle")
fig.colorbar(pos)
ax.grid()

for i in range(n_sites-1):
    ax.axvline(x = (i+1)*2-0.5,color='black')
    ax.axhline(y = (i+1)*2-0.5,color='black')

plt.tight_layout()

n_states = 2*n_sites

n_block = n_states+4
ham_eigs,ham_vecs = np.linalg.eigh(ham_bdg)

# Lead properties
tL = 30
DeltaL = 1.0
TL = 0.01

tR = 30
DeltaR = 1.0
TR = 0.01


# Define the self-energy
tSL = 5.3
tSR = 5.3

bias = 0.5

muL = 0.0
muR = 0.0
ieta = 1.0e-3j

cutoff_below = -10
cutoff_above = 4

#base_freq = cutoff_above-cutoff_below
base_freq = bias
n_freqs = math.ceil((cutoff_above-cutoff_below)/base_freq)
print(n_freqs)



sigmas = []
sigma0 = np.zeros([n_block,n_block],dtype=complex)
sigma0[-4,-2] = tSR
sigma0[-3,-1] = -np.conj(tSR)
sigma0[-2,-4] = np.conj(tSR)
sigma0[-1,-3] = -tSR
#sigma0[0,2] = np.conj(tSL)
#sigma0[1,3] = -tSL
#sigma0[2,0] = tSL
#sigma0[3,1] = -np.conj(tSL)
#sigmas.append((sigma0,0))

print(sigma0)

sigma1 = np.zeros([n_block,n_block],dtype=complex)
sigma1[2,0] = tSL
sigma1[1,3] = -tSL
sigmam1 = np.zeros([n_block,n_block],dtype=complex)
sigmam1[0,2] = np.conj(tSL)
sigmam1[3,1] = -np.conj(tSL)
sigmas.append((sigma0,0))
sigmas.append((sigma1,1))
sigmas.append((sigmam1,-1))


SigmaR = assemble_SigmaR(sigmas,n_freqs)

fig,ax = plt.subplots()

pos = ax.matshow(np.real(SigmaR),interpolation='nearest')
ax.set_xticks(np.arange(0.5,len(SigmaR)-0.5,1))
ax.set_yticks(np.arange(0.5,len(SigmaR)-0.5,1))
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.colorbar(pos)
ax.grid()

for i in range(n_freqs):
    if i != n_freqs-1:
        ax.axvline(x = (i+1)*n_block-0.5,color='black',lw=2)
        ax.axhline(y = (i+1)*n_block-0.5,color='black',lw=2)
    for j in range(n_sites+2-1):
        ax.axvline(x = i*n_block + (j+1)*2-0.5,color='black',lw=1)
        ax.axhline(y = i*n_block + (j+1)*2-0.5,color='black',lw=1)

plt.tight_layout()


n_Es = 300
#Es = np.linspace(cutoff_below,cutoff_above,n_Es)
Es = np.linspace(-0.1,0.1,n_Es)
Es = np.append(Es, np.linspace(-0.1+base_freq,0.1+base_freq,n_Es))
Es = np.append(Es, np.linspace(-0.1+2*base_freq,0.1+2*base_freq,n_Es))
Es = np.append(Es, np.linspace(-0.1-base_freq,0.1-base_freq,n_Es))
n_Es = Es.shape[0]

#Es = [-2*base_freq,-base_freq,0,base_freq,2*base_freq]
#n_Es = len(Es)
Gln0 = np.zeros([n_Es,n_block,n_block],dtype=complex)
Gln1 = np.zeros([n_Es,n_block,n_block],dtype=complex)
Glnm1 = np.zeros([n_Es,n_block,n_block],dtype=complex)
#Glnns = np.zeros([n_Es,SigmaR.shape[0],SigmaR.shape[1]],dtype=complex)

for n in range(n_Es):
    print(n)
    GR,Gl = Greens_freq_nn(Es[n],ham_eigs,ham_vecs,SigmaR,base_freq,cutoff_below,cutoff_above,muL,DeltaL,tL,TL, DeltaR,tR,TR,ieta)
    Gln0[n,:,:] = get_Gln(Gl,0,n_freqs,n_block)
    Gln1[n,:,:] = get_Gln(Gl,1,n_freqs,n_block)
    Glnm1[n,:,:] = get_Gln(Gl,-1,n_freqs,n_block)

#Glnns_reshape = Glnns.reshape([n_Es,n_freqs,n_block,n_freqs,n_block])
#print(Glnns_reshape.shape)
#Glnns_orig = Glnns_reshape.reshape([n_Es,n_freqs*n_block,n_freqs*n_block])
#
#print(Glnns_reshape.shape)
#Gln0 = np.trace(Glnns_reshape,offset=0,axis1=1,axis2=3,dtype=complex)
#Gln1 = np.trace(Glnns_reshape,offset=1,axis1=1,axis2=3,dtype=complex)
#Glnm1 = np.trace(Glnns_reshape,offset=-1,axis1=1,axis2=3,dtype=complex)




print(np.real(np.diag(np.sum(-1.0j*Gln0,axis=0)*base_freq/(np.pi*n_Es))))

fix,ax = plt.subplots()
#ax.plot(Es,np.imag(Gln0_alt[:,2,2]))
ax.plot(Es,np.real(Gln1[:,0,2]),label="p02")
ax.plot(Es,np.real(Gln1[:,2,0]),label="p20")
ax.plot(Es,np.real(Glnm1[:,0,2]),label="m02")
ax.plot(Es,np.real(Glnm1[:,2,0]),label="m20")
plt.legend()

plt.show()




            




