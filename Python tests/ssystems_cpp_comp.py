import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit


np.set_printoptions(threshold=sys.maxsize,precision=3)

def fd(E,T):
    if np.abs(T) < 1e-5:
        if E > 0.0:
            return 0.0
        else:
            return 1.0

    return 1.0/(1.0+np.exp(E/T))

tL = 30
tR = 30

# assemble single particle Hamiltonian
#eA = -2.0
#eB = -2.0
#eA = 3.236
#eB = 3.236
#eA = 1.414
#eB = 1.414
eA = 0.0
eB = 0.0
tAA = -1.0
tAB = np.sqrt(2.0)*tAA

sigmaz = np.matrix([[1.0,0.0],[0.0,-1.0]],dtype=complex)

Ncells = 4
Norbs = 2
on_site_energies = np.matrix([[eA,0],[0,eB]],dtype=complex)
intra_cell_hopping = np.matrix([[0,tAB],[np.conj(tAB),0]], dtype=complex)
inter_cell_hopping = np.matrix([[tAA,tAB],[0.0,0.0]], dtype=complex)
deleted_sites_m = []
deleted_sites_p = [1]

Nsites = Ncells*Norbs-len(deleted_sites_m)-len(deleted_sites_p)

H0 = np.zeros((Ncells*Norbs,Ncells*Norbs),dtype = complex)

for i in range(Ncells):
    H0[i*Norbs:(i+1)*Norbs,i*Norbs:(i+1)*Norbs] = on_site_energies +intra_cell_hopping
    if i > 0:
        H0[i*Norbs:(i+1)*Norbs,(i-1)*Norbs:(i)*Norbs] = inter_cell_hopping
    if i < Ncells-1:
        H0[i*Norbs:(i+1)*Norbs,(i+1)*Norbs:(i+2)*Norbs] = np.conj(inter_cell_hopping.T)

for i in deleted_sites_m:
    H0 = np.delete(H0,i,0)
    H0 = np.delete(H0,i,1)

for i in deleted_sites_p:
    H0 = np.delete(H0,-Norbs+i,0)
    H0 = np.delete(H0,-Norbs+i,1)


I0 = np.eye(Nsites,dtype=complex)
IBdG = np.eye(2*Nsites,dtype=complex)
IBdGl = np.eye(2*Nsites+4,dtype=complex)
DeltaS = np.zeros(Nsites,dtype=complex)
HartreeS = np.zeros(Nsites,dtype=complex)

HBdG = np.zeros([2*Nsites,2*Nsites],dtype=complex)
for i in range(Nsites):
    HBdG[2*i,2*i] = H0[i,i] + HartreeS[i]
    HBdG[2*i+1,2*i+1] = -np.conj(H0[i,i]) - HartreeS[i]
    HBdG[2*i,2*i+1] = DeltaS[i]
    HBdG[2*i+1,2*i] = np.conj(DeltaS[i])
    for j in range(Nsites):
        if j != i:
            HBdG[2*i,2*j] = H0[i,j] 
            HBdG[2*i+1,2*j+1] = -np.conj(H0[i,j])


tSL = np.sqrt(tL)
tLS = np.sqrt(tL)
tSR = np.sqrt(tL)
tRS = np.sqrt(tL)


DeltaL = 1.0
DeltaR = 1.0
DeltaS = 0.0
ieta = 1.0j*1.0e-3
T = 0

E_cutoff = 10*DeltaL

Nsl = Nsites+2

HSL = np.zeros((2*Nsites,2),dtype=complex)
HSL[0,0] = tSL
HSL[1,1] = tLS



#sys.exit("Setups")

HLS01 = np.zeros((2*Nsl,2*Nsl),dtype=complex)
HLS10 = np.zeros((2*Nsl,2*Nsl),dtype=complex)
HLS00 = np.zeros((2*Nsl,2*Nsl),dtype=complex)
HRS01 = np.zeros((2*Nsl,2*Nsl),dtype=complex)
HRS10 = np.zeros((2*Nsl,2*Nsl),dtype=complex)
HRS00 = np.zeros((2*Nsl,2*Nsl),dtype=complex)

HLS10[0,2] = tLS
HLS01[1,3] = -tSL
HLS01[2,0] = tSL
HLS10[3,1] = -tLS

HRS10[-4,-2] = tSR
HRS01[-3,-1] = -tRS
HRS01[-2,-4] = tRS
HRS10[-1,-3] = -tSR

#H00[2,4] = tSR
#H00[3,5] = -tRS
#H00[4,2] = tRS
#H00[5,3] = -tSR


I2 = np.eye(2,dtype=complex)


#sys.exit("Setups")
    
zero22 = np.zeros([2,2],dtype=complex)
zero2Ns = np.zeros([2,2*Nsites],dtype=complex)

#ISL = np.zeros(2*n_cutoff+1, dtype = complex)
#ns = np.zeros(2*n_cutoff+1, dtype = complex)

n_Vs = 50
Vs = np.linspace(0.3*DeltaL,2.5*DeltaL,n_Vs)
#Vs = np.array([0.167,0.2,0.25,0.33,0.5,1.0],dtype=complex)


n_Es = 100
Is1 = np.zeros(n_Vs,dtype = complex)
nums = np.zeros(n_Vs,dtype = complex)

for j in range(n_Vs):
    #start = timeit.default_timer()
    #ISL = np.zeros(2*n_cutoff+1, dtype = complex)
    V = Vs[j]
    n_lower_cutoff = 2*math.ceil(E_cutoff/V)

    n_upper_cutoff = n_lower_cutoff

    # note that we want V spaced frequencies with (n+0.5)V frequency for each 
    # so also the zero frequency is coundted twice, for 0 and 0.5V.
    # This is done to make the correspondence to the case of lead self-energy
    # in the closed system clearer, where frequencies only V apart are considered
    #num_freqs= n_lower_cutoff + n_upper_cutoff+2
    num_freqs= n_lower_cutoff + n_upper_cutoff+1

    # Half the number of frequency for even-odd separation
    half_freqs = math.floor(num_freqs/2)

    # Useful identity matrices
    I = np.eye(2*Nsl*num_freqs,dtype=complex)
    sigmaz_nn_w = np.kron(np.eye(num_freqs,dtype=complex),sigmaz);

    # Energies to be used
    Es = np.linspace(0,abs(V/2),n_Es)

    # Assemble left and right Sigma, that is, the hopping hamiltonian between the
    # leads and the system
    SigmaLS = np.zeros((num_freqs*2*Nsl,num_freqs*2*Nsl), dtype = complex)
    SigmaRS = np.zeros((num_freqs*2*Nsl,num_freqs*2*Nsl), dtype = complex)
    Sigma = np.zeros((num_freqs*2*Nsl,num_freqs*2*Nsl), dtype = complex)
    for i in range(num_freqs-1):
        SigmaLS[2*Nsl*i:2*Nsl*(i+1),2*Nsl*(i+1):2*Nsl*(i+2)] = HLS01
        SigmaLS[2*Nsl*(i+1):2*Nsl*(i+2),2*Nsl*(i):2*Nsl*(i+1)] = HLS10
        SigmaLS[2*Nsl*(i):2*Nsl*(i+1),2*Nsl*(i):2*Nsl*(i+1)] = HLS00
        SigmaRS[2*Nsl*i:2*Nsl*(i+1),2*Nsl*(i+1):2*Nsl*(i+2)] = HRS01
        SigmaRS[2*Nsl*(i+1):2*Nsl*(i+2),2*Nsl*(i):2*Nsl*(i+1)] = HRS10
        SigmaRS[2*Nsl*(i):2*Nsl*(i+1),2*Nsl*(i):2*Nsl*(i+1)] = HRS00
    SigmaLS[2*Nsl*(num_freqs-1):2*Nsl*(num_freqs),2*Nsl*(num_freqs-1):2*Nsl*(num_freqs)] = HLS00
    SigmaRS[2*Nsl*(num_freqs-1):2*Nsl*(num_freqs),2*Nsl*(num_freqs-1):2*Nsl*(num_freqs)] = HRS00

    Sigma = SigmaLS+SigmaRS

    print("V = "+str(V))
    for i in range(n_Es):
        E = Es[i]
        # Calculate non-perturbed Green's functions
        gR = np.zeros((2*Nsl*(num_freqs),2*Nsl*(num_freqs)), dtype = complex) 
        gl = np.zeros((2*Nsl*(num_freqs),2*Nsl*(num_freqs)), dtype = complex)
        for k in range(-n_lower_cutoff,n_upper_cutoff+1,1):
            # Definition of the lead self-energy
            # self-energy couples energies V apart due to the 
            # pair Green's functions in the leads
    
            #print('E',E+k*V/2)
            # Retarded 
            gRLL_const = 1.0/tL*1.0/np.sqrt(abs(DeltaL)**2-(E+k*V/2+ieta)**2)
            gRRR_const = 1.0/tR*1.0/np.sqrt(abs(DeltaR)**2-(E+k*V/2+ieta)**2)
                                                           
            gRLL11 = gRLL_const*(-E-k*V/2-ieta)
            gRLL22 = gRLL_const*(-E-k*V/2-ieta)
            
            gRRR11 = gRRR_const*(-E-k*V/2-ieta)
            gRRR22 = gRRR_const*(-E-k*V/2-ieta)

            gRLL12 = gRLL_const*(DeltaL)
            gRLL21 = gRLL_const*(np.conj(DeltaL))

            gRRR12 = gRRR_const*(DeltaR)
            gRRR21 = gRRR_const*(np.conj(DeltaR))

            # system Green's functions
            #gRSS11 = 1.0/((E+k*V/2+ieta)-eQD)
            #gRSS22 = 1.0/((E+k*V/2+ieta)+eQD)
            gRSSinv = (E+k*V/2+ieta)*IBdG-HBdG
            gRSS = np.linalg.solve(gRSSinv,IBdG)

            # assemble
            gRLL = np.matrix([[gRLL11,gRLL12],[gRLL21,gRLL22]], dtype = complex)
            gRRR = np.matrix([[gRRR11,gRRR12],[gRRR21,gRRR22]], dtype = complex)
            #gRSS = np.matrix([[gRSS11,0.0],[0.0,gRSS22]], dtype = complex)


            gRnn = np.block([[gRLL,zero2Ns,zero22],[zero2Ns.T,gRSS,zero2Ns.T],[zero22,zero2Ns,gRRR]])
            gAnn = gRnn.conj().T
            glnn = (gAnn-gRnn)*fd(E+k*V/2,T)
    
            n = k+n_lower_cutoff # matrix index

            gR[2*Nsl*n:2*Nsl*(n+1),2*Nsl*n:2*Nsl*(n+1)] = gRnn
            gl[2*Nsl*n:2*Nsl*(n+1),2*Nsl*n:2*Nsl*(n+1)] = glnn
    

        GR = np.linalg.solve(I-np.matmul(gR,Sigma),gR)
        GA = GR.conj().T
        IpGRSigmaR = I+np.matmul(GR,Sigma);
        #Gl = np.matmul(np.matmul((I+np.matmul(GR,Sigma)),gl),(I+np.matmul(Sigma,GA)))
        Gl = np.matmul(np.matmul(IpGRSigmaR,gl),IpGRSigmaR.conjugate().T)

        print("E=",E,"Gl calculation done",E)
        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #ax1.matshow(np.real(Sigma),interpolation='nearest')
        #ax1.set_xticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_yticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_title("Sigma py")
        #ax1.grid()

        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #ax1.matshow(np.imag(GR),interpolation='nearest')
        #ax1.set_xticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_yticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_title("GR py")
        #ax1.grid()

        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #ax1.matshow(np.imag(Gl),interpolation='nearest')
        #ax1.set_xticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_yticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_title("Gl py")
        #ax1.grid()

        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #ax1.matshow(np.imag(gR),interpolation='nearest')
        #ax1.set_xticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_yticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_title("gR py")
        #ax1.grid()

        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #ax1.matshow(np.imag(gl),interpolation='nearest')
        #ax1.set_xticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_yticks(np.arange(1.5,len(GR)-1.5,2))
        #ax1.set_title("gl py")
        #ax1.grid()

        #plt.show()


        # Way 1 ot calculate current
        #SigmaGlSL = np.matmul(Sigma[0::2*Nsl,2::2*Nsl],Gl[2::2*Nsl,0::2*Nsl])-np.matmul(Sigma[2::2*Nsl,0::2*Nsl],Gl[0::2*Nsl,2::2*Nsl])
        SigmaGlSL = np.matmul(Sigma[0::2*Nsl,2::2*Nsl],Gl[2::2*Nsl,0::2*Nsl])
        nums[j] = 2.0*np.real(-1.0j*np.trace(Gl[2::2*Nsl,2::2*Nsl]));
        Is1[j] += -4*np.real(np.trace(SigmaGlSL))

    Is1[j] *= (V/2)/(n_Es-1)*1.0/(2*np.pi)
    nums[j] *= (V/2)/(n_Es-1)*1.0/(2*np.pi)
    
    #fix, ax1 = plt.subplots()
    #ax1.plot(ts,It)
    #ax1.set_xlabel("t")
    #ax1.set_ylabel("Current")
    #ax1.set_title("V = " + str(V))
    #ISL *= V/(n_Es-1)*1.0/2*(np.pi)
    #print("V = " + str(V))
    #print(ISL)
    #Is[j] = ISL[n_cutoff]

        
#    ns *= V/(n_Es-1)*1/(4*np.pi**2)
#    for i in range(2*n_cutoff+1):
#        print(str(-n_cutoff+i)+" "+str(ISL[i]))
#    for i in range(2*n_cutoff+1):
#        print(str(-n_cutoff+i)+" "+str(ns[i]))
#
#nt = np.zeros(n_ts,dtype=complex)
#
#fix, ax2 = plt.subplots()
#ax2.plot(ts,nt)
#
#
#w = np.fft.fft(It)
#idx = np.argmax(np.abs(w))
#freqs = np.fft.fftfreq(len(w))
#print(freqs.min(),freqs.max())
#freq = freqs[idx]
#
#
#print(freq*1000/10)


fig =  plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(Vs/DeltaL,np.real(Is1)/DeltaL)
ax1.set_xlabel("Voltage/$\Delta$")
ax1.set_ylabel("Current/$\Delta$")
ax1.set_title("Current")

ax2 = fig.add_subplot(122)
ax2.plot(Vs/DeltaL,np.real(nums))
ax2.set_xlabel("Voltage/$\Delta$")
ax2.set_ylabel("Particle Number")
ax2.set_title("Particle Number")



plt.show()
