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

Ncells = 1
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

n_Vs = 30
Vs = np.linspace(0.3*DeltaL,2.5*DeltaL,n_Vs)
#Vs = np.array([0.167,0.2,0.25,0.33,0.5,1.0],dtype=complex)


n_Es = 100
Is1 = np.zeros(n_Vs,dtype = complex)
Is2 = np.zeros(n_Vs,dtype = complex)

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
    num_freqs= n_lower_cutoff + n_upper_cutoff+2

    # Half the number of frequency for even-odd separation
    half_freqs = math.floor(num_freqs/2)

    # Useful identity matrices
    I = np.eye(2*Nsl*num_freqs,dtype=complex)
    Ihalf = np.eye(Nsl*num_freqs,dtype=complex)
    Isys = np.eye(2*Nsites*num_freqs,dtype=complex)
    Isystemodd = np.eye(2*half_freqs*Nsites,dtype=complex)
    Isystemeven = np.eye(2*half_freqs*Nsites,dtype=complex)

    sigmaz_nn = np.kron(np.eye(half_freqs,dtype=complex),sigmaz);
    sigmaz_nn_w = np.kron(np.eye(num_freqs,dtype=complex),sigmaz);

    odd_idx = [2*Nsl*2*n+2*j for n in range(half_freqs) for j in range(Nsl)] + [2*Nsl*2*n+2*j+1 for n in range(half_freqs) for j in range(Nsl)]
    even_idx = [2*Nsl*(2*n+1)+2*j for n in range(half_freqs) for j in range(Nsl)] + [2*Nsl*(2*n+1)+2*j+1 for n in range(half_freqs) for j in range(Nsl)]

    sys_idx_whole = [2*Nsl*n+2*(m+1) for n in range(num_freqs) for m in range(Nsites)] + [2*Nsl*n+2*(m+1)+1 for n in range (num_freqs) for m in range(Nsites)]
    leadL_idx_whole = [2*Nsl*n for n in range(num_freqs)] + [2*Nsl*n+1 for n in range(num_freqs)]
    leadR_idx_whole = [2*Nsl*n+2*Nsl-2 for n in range(num_freqs) ] + [2*Nsl*n+2*Nsl-1 for n in range(num_freqs)]

    sys_idx_half = [2*Nsl*n+2*(m+1) for n in range(half_freqs) for m in range(Nsites)] + [2*Nsl*n+2*(m+1)+1 for n in range (half_freqs) for m in range(Nsites)]
    leadL_idx_half = [2*Nsl*n for n in range(half_freqs)] + [2*Nsl*n+1 for n in range(half_freqs)]
    leadR_idx_half = [2*Nsl*n+2*Nsl-2 for n in range(half_freqs) ] + [2*Nsl*n+2*Nsl-1 for n in range(half_freqs)]

    odd_idx_sys = [2*Nsites*2*n+2*j for n in range(half_freqs) for j in range(Nsites)] + [2*Nsites*2*n+2*j+1 for n in range(half_freqs) for j in range(Nsites)]
    even_idx_sys = [2*Nsites*(2*n+1)+2*j for n in range(half_freqs) for j in range(Nsites)] + [2*Nsites*(2*n+1)+2*j+1 for n in range(half_freqs) for j in range(Nsites)]

    odd_idx_leadL = [2*2*n for n in range(half_freqs)] + [2*2*n+1 for n in range(half_freqs)]
    even_idx_leadL = [2*(2*n+1) for n in range(half_freqs)] + [2*(2*n+1)+1 for n in range(half_freqs)]

    LC_idx_odd_sys = [2*Nsites*n for n in range(half_freqs)] + [2*Nsites*n+1 for n in range(half_freqs)]
    LC_idx_even_sys = [2*Nsites*n for n in range(half_freqs)] + [2*Nsites*n+1 for n in range(half_freqs)]

    odd_idx.sort()
    even_idx.sort()
    odd_idx_sys.sort()
    even_idx_sys.sort()
    odd_idx_leadL.sort()
    even_idx_leadL.sort()
    sys_idx_whole.sort()
    sys_idx_half.sort()
    leadL_idx_whole.sort()
    leadR_idx_whole.sort()
    leadL_idx_half.sort()
    leadR_idx_half.sort()


    #print("odd_idx")
    #print(odd_idx)
    #print("even_idx")
    #print(even_idx)
    #print("odd_idx_sys")
    #print(odd_idx_sys)
    #print("even_idx_sys")
    #print(even_idx_sys)
    #print("sys_idx_whole")
    #print(sys_idx_whole)
    #print("sys_idx_half")
    #print(sys_idx_half)
    #print("leadL_idx_whole")
    #print(leadL_idx_whole)
    #print("leadR_idx_whole")
    #print(leadR_idx_whole)
    #print("leadL_idx_half")
    #print(leadL_idx_half)
    #print("leadR_idx_half")
    #print(leadR_idx_half)

    # Reordering to separate odd and even frequency blocks
    reorder_idx = odd_idx + even_idx

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
        for k in range(-n_lower_cutoff,n_upper_cutoff+2,1):
            # Definition of the lead self-energy
            # self-energy couples energies V apart due to the 
            # pair Green's functions in the leads
    
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
    


        # Reorder to have odd and even blocks separately
        #gR_reorder = gR[np.ix_(reorder_idx,reorder_idx)]
        #gl_reorder = gl[np.ix_(reorder_idx,reorder_idx)]
        #Sigma_reorder = Sigma[np.ix_(reorder_idx,reorder_idx)]

        # Solve GR in reordered basis
        #GR_reorder = np.linalg.solve((I-np.matmul(gR_reorder,Sigma_reorder)),gR_reorder)
        #GR_alt_orig = np.asmatrix(np.zeros(GR_reorder.shape,dtype=complex))
        #GR_alt_orig[np.ix_(reorder_idx,reorder_idx)] = GR_reorder

        ## odd,even components
        gRodd = gR[np.ix_(odd_idx,odd_idx)]
        glodd = gl[np.ix_(odd_idx,odd_idx)]
        gReven = gR[np.ix_(even_idx,even_idx)]
        gleven = gl[np.ix_(even_idx,even_idx)]
        Sigmaoddeven = Sigma[np.ix_(odd_idx,even_idx)]
        Sigmaevenodd = Sigma[np.ix_(even_idx,odd_idx)]

        SigmaLSoddeven = SigmaLS[np.ix_(odd_idx,even_idx)]
        SigmaLSevenodd = SigmaLS[np.ix_(even_idx,odd_idx)]

        ### odd,even Sigmas for the system
        SigmaR0Lodd = np.matmul(np.matmul(Sigmaoddeven,gReven),Sigmaevenodd)
        SigmaA0Lodd = SigmaR0Lodd.conj().T
        Sigmal0Lodd = np.matmul(np.matmul(Sigmaoddeven,gleven),Sigmaevenodd)
        SigmaR0Leven = np.matmul(np.matmul(Sigmaevenodd,gRodd),Sigmaoddeven)
        SigmaA0Leven = SigmaR0Leven.conj().T
        Sigmal0Leven = np.matmul(np.matmul(Sigmaevenodd,glodd),Sigmaoddeven)
        
        # Odd and even Sigma caused by the left lead only
        SigmaR0LSodd = np.matmul(np.matmul(SigmaLSoddeven,gReven),SigmaLSevenodd)
        SigmaA0LSodd = SigmaR0LSodd.conj().T
        Sigmal0LSodd = np.matmul(np.matmul(SigmaLSoddeven,gleven),SigmaLSevenodd)
        SigmaR0LSeven = np.matmul(np.matmul(SigmaLSevenodd,gRodd),SigmaLSoddeven)
        SigmaA0LSeven = SigmaR0LSeven.conj().T
        Sigmal0LSeven = np.matmul(np.matmul(SigmaLSevenodd,glodd),SigmaLSoddeven)

        # Odd system self-energy
        SigmaRSSodd1 = SigmaR0Lodd[np.ix_(sys_idx_half,sys_idx_half)]
        SigmaASSodd1 = SigmaA0Lodd[np.ix_(sys_idx_half,sys_idx_half)]
        SigmalSSodd1 = Sigmal0Lodd[np.ix_(sys_idx_half,sys_idx_half)]

        # Odd system self-energy from left lead (for current calculation)
        SigmaRSSLodd1 = SigmaR0LSodd[np.ix_(sys_idx_half,sys_idx_half)]
        SigmaASSLodd1 = SigmaA0LSodd[np.ix_(sys_idx_half,sys_idx_half)]
        SigmalSSLodd1 = Sigmal0LSodd[np.ix_(sys_idx_half,sys_idx_half)]

        # Even system self-energy
        SigmaRSSeven1 = SigmaR0Leven[np.ix_(sys_idx_half,sys_idx_half)]
        SigmaASSeven1 = SigmaA0Leven[np.ix_(sys_idx_half,sys_idx_half)]
        SigmalSSeven1 = Sigmal0Leven[np.ix_(sys_idx_half,sys_idx_half)]

        # Even system self-energy from left lead (for current calculation)
        SigmaRSSLeven1 = SigmaR0LSeven[np.ix_(sys_idx_half,sys_idx_half)]
        SigmaASSLeven1 = SigmaA0LSeven[np.ix_(sys_idx_half,sys_idx_half)]
        SigmalSSLeven1 = Sigmal0LSeven[np.ix_(sys_idx_half,sys_idx_half)]

        gRSSodd = gRodd[np.ix_(sys_idx_half,sys_idx_half)]
        gRLLeven = gReven[np.ix_(leadL_idx_half,leadL_idx_half)]
        gRRReven = gReven[np.ix_(leadR_idx_half,leadR_idx_half)]

        gRSSeven = gReven[np.ix_(sys_idx_half,sys_idx_half)]
        gRLLodd = gRodd[np.ix_(leadL_idx_half,leadL_idx_half)]
        gRRRodd = gRodd[np.ix_(leadR_idx_half,leadR_idx_half)]

        glSSodd = glodd[np.ix_(sys_idx_half,sys_idx_half)]
        glLLeven = gleven[np.ix_(leadL_idx_half,leadL_idx_half)]
        glRReven = gleven[np.ix_(leadR_idx_half,leadR_idx_half)]

        glSSeven = gleven[np.ix_(sys_idx_half,sys_idx_half)]
        glLLodd = glodd[np.ix_(leadL_idx_half,leadL_idx_half)]
        glRRodd = glodd[np.ix_(leadR_idx_half,leadR_idx_half)]

        # Calculate odd Sigma for the system from precalculated fromula
        #SigmaRSSLodd2 = np.zeros((2*Nsites*half_freqs,2*Nsites*half_freqs),dtype=complex)
        #SigmaRSSRodd2 = np.zeros((2*Nsites*half_freqs,2*Nsites*half_freqs),dtype=complex)
        #SigmalSSLodd2 = np.zeros((2*Nsites*half_freqs,2*Nsites*half_freqs),dtype=complex)
        #SigmalSSRodd2 = np.zeros((2*Nsites*half_freqs,2*Nsites*half_freqs),dtype=complex)
        #for n in range(half_freqs):
        #    m_left = 2*Nsites*n
        #    m_right = 2*Nsites*(n+1)-2
        #    SigmaRSSLodd2[m_left,m_left] = tSL*tLS*gRLLeven[2*n,2*n] 
        #    SigmaRSSRodd2[m_right+1,m_right+1] = tSR*tRS*gRRReven[2*n+1,2*n+1] 
        #    SigmalSSLodd2[m_left,m_left] = tSL*tLS*glLLeven[2*n,2*n] 
        #    SigmalSSRodd2[m_right+1,m_right+1] = tSR*tRS*glRReven[2*n+1,2*n+1] 
        #    if n>0:
        #        SigmaRSSRodd2[m_right,m_right] += tSR*tRS*gRRReven[2*(n-1),2*(n-1)] 
        #        SigmaRSSLodd2[m_left+1,m_left+1] += tSL*tLS*gRLLeven[2*(n-1)+1,2*(n-1)+1] 
        #        SigmalSSRodd2[m_right,m_right] += tSR*tRS*glRReven[2*(n-1),2*(n-1)] 
        #        SigmalSSLodd2[m_left+1,m_left+1] += tSL*tLS*glLLeven[2*(n-1)+1,2*(n-1)+1] 

        #    if n > 0:
        #        SigmaRSSRodd2[m_right,m_right-2*Nsites+1] +=-tSR*tRS*gRRReven[2*(n-1),2*(n-1)+1]
        #        SigmaRSSLodd2[m_left+1,m_left-2*Nsites] +=-tSL*tLS*gRLLeven[2*(n-1)+1,2*(n-1)]
        #        SigmalSSRodd2[m_right,m_right-2*Nsites+1] +=-tSR*tRS*glRReven[2*(n-1),2*(n-1)+1]
        #        SigmalSSLodd2[m_left+1,m_left-2*Nsites] +=-tSL*tLS*glLLeven[2*(n-1)+1,2*(n-1)]
        #    if n < half_freqs-1:
        #        SigmaRSSLodd2[m_left,m_left+2*Nsites+1] +=-tSL*tSL*gRLLeven[2*n,2*n+1]
        #        SigmaRSSRodd2[m_right+1,m_right+2*Nsites] +=-tSR*tSR*gRRReven[2*n+1,2*n]
        #        SigmalSSLodd2[m_left,m_left+2*Nsites+1] +=-tSL*tSL*glLLeven[2*n,2*n+1]
        #        SigmalSSRodd2[m_right+1,m_right+2*Nsites] +=-tSR*tSR*glRReven[2*n+1,2*n]
        #SigmaASSLodd2 = SigmaRSSLodd2.conj().T
        #SigmaRSSodd2 = SigmaRSSLodd2+SigmaRSSRodd2
        #SigmalSSodd2 = SigmalSSLodd2+SigmalSSRodd2

        ## Calculate even Sigma for the system from precalculated fromula
        #SigmaRSSLeven2 = np.zeros((2*Nsites*half_freqs,2*Nsites*half_freqs),dtype=complex)
        #SigmaRSSReven2 = np.zeros((2*Nsites*half_freqs,2*Nsites*half_freqs),dtype=complex)
        #SigmalSSLeven2 = np.zeros((2*Nsites*half_freqs,2*Nsites*half_freqs),dtype=complex)
        #SigmalSSReven2 = np.zeros((2*Nsites*half_freqs,2*Nsites*half_freqs),dtype=complex)
        #for n in range(half_freqs):
        #    m_left = 2*Nsites*n
        #    m_right = 2*Nsites*(n+1)-2
        #    SigmaRSSLeven2[m_left,m_left] = tSL*tLS*gRLLeven[2*n,2*n] 
        #    SigmaRSSReven2[m_right+1,m_right+1] = tSR*tRS*gRRReven[2*n+1,2*n+1] 
        #    SigmalSSLeven2[m_left,m_left] = tSL*tLS*glLLeven[2*n,2*n] 
        #    SigmalSSReven2[m_right+1,m_right+1] = tSR*tRS*glRReven[2*n+1,2*n+1] 
        #    if n>0:
        #        SigmaRSSReven2[m_right,m_right] += tSR*tRS*gRRReven[2*(n-1),2*(n-1)] 
        #        SigmaRSSLeven2[m_left+1,m_left+1] += tSL*tLS*gRLLeven[2*(n-1)+1,2*(n-1)+1] 
        #        SigmalSSReven2[m_right,m_right] += tSR*tRS*glRReven[2*(n-1),2*(n-1)] 
        #        SigmalSSLeven2[m_left+1,m_left+1] += tSL*tLS*glLLeven[2*(n-1)+1,2*(n-1)+1] 

        #    if n > 0:
        #        SigmaRSSReven2[m_right,m_right-2*Nsites+1] +=-tSR*tRS*gRRReven[2*(n-1),2*(n-1)+1]
        #        SigmaRSSLeven2[m_left+1,m_left-2*Nsites] +=-tSL*tLS*gRLLeven[2*(n-1)+1,2*(n-1)]
        #        SigmalSSReven2[m_right,m_right-2*Nsites+1] +=-tSR*tRS*glRReven[2*(n-1),2*(n-1)+1]
        #        SigmalSSLeven2[m_left+1,m_left-2*Nsites] +=-tSL*tLS*glLLeven[2*(n-1)+1,2*(n-1)]
        #    if n < half_freqs-1:
        #        SigmaRSSLeven2[m_left,m_left+2*Nsites+1] +=-tSL*tSL*gRLLeven[2*n,2*n+1]
        #        SigmaRSSReven2[m_right+1,m_right+2*Nsites] +=-tSR*tSR*gRRReven[2*n+1,2*n]
        #        SigmalSSLeven2[m_left,m_left+2*Nsites+1] +=-tSL*tSL*glLLeven[2*n,2*n+1]
        #        SigmalSSReven2[m_right+1,m_right+2*Nsites] +=-tSR*tSR*glRReven[2*n+1,2*n]
        #SigmaASSLeven2 = SigmaRSSLeven2.conj().T
        #SigmaRSSeven2 = SigmaRSSLeven2+SigmaRSSReven2
        #SigmalSSeven2 = SigmalSSLeven2+SigmalSSReven2


        ## Gather all to single matrix
        #GR_alt_order = np.block([[GRodd,GRoddeven],[GRevenodd,GReven]])

        GR = np.linalg.solve(I-np.matmul(gR,Sigma),gR)
        GA = GR.conj().T
        Gl = np.matmul(np.matmul((I+np.matmul(GR,Sigma)),gl),(I+np.matmul(Sigma,GA)))

        print("V=",V,"Gl calculation done",E)


        # Way 1 ot calculate current
        #SigmaGlSL = np.matmul(Sigma[0::2*Nsl,2::2*Nsl],Gl[2::2*Nsl,0::2*Nsl])-np.matmul(Sigma[2::2*Nsl,0::2*Nsl],Gl[0::2*Nsl,2::2*Nsl])
        #Is1[j] += -2*np.real(np.trace(SigmaGlSL))

        cur_matrix_1 = np.matmul(Sigma[np.ix_(leadL_idx_whole,sys_idx_whole)],Gl[np.ix_(sys_idx_whole,leadL_idx_whole)])
        cur_matrix_1 -= np.matmul(Sigma[np.ix_(sys_idx_whole,leadL_idx_whole)],Gl[np.ix_(leadL_idx_whole,sys_idx_whole)])
        cur_matrix_1 = np.matmul(sigmaz_nn_w,cur_matrix_1)

        Is1[j] += -np.trace(cur_matrix_1)

        #cur_matrix_odd1 = np.matmul(sigmaz_nn,cur_matrix_1[np.ix_(odd_idx_leadL,odd_idx_leadL)])
        #cur_matrix_even1 = np.matmul(sigmaz_nn,cur_matrix_1[np.ix_(even_idx_leadL,even_idx_leadL)])


        #GRodd = GR[np.ix_(odd_idx,odd_idx)]
        #GReven = GR[np.ix_(even_idx,even_idx)]
        #GAodd = GA[np.ix_(odd_idx,odd_idx)]
        #GAeven = GA[np.ix_(even_idx,even_idx)]

        #Glodd = Gl[np.ix_(odd_idx,odd_idx)]
        #Gleven = Gl[np.ix_(even_idx,even_idx)]

        #GRSSodd1 = GRodd[np.ix_(sys_idx_half,sys_idx_half)]
        #GRSSeven1 = GReven[np.ix_(sys_idx_half,sys_idx_half)]
        #GASSodd1 = GAodd[np.ix_(sys_idx_half,sys_idx_half)]
        #GASSeven1 = GAeven[np.ix_(sys_idx_half,sys_idx_half)]
        #GlSSodd1 = Glodd[np.ix_(sys_idx_half,sys_idx_half)]
        #GlSSeven1 = Gleven[np.ix_(sys_idx_half,sys_idx_half)]


        #SigmalSS = np.matmul(np.matmul(Sigma[np.ix_(sys_idx_whole,leadL_idx_whole)],gl[np.ix_(leadL_idx_whole,leadL_idx_whole)]),Sigma[np.ix_(leadL_idx_whole,sys_idx_whole)])
        #SigmalSS += np.matmul(np.matmul(Sigma[np.ix_(sys_idx_whole,leadR_idx_whole)],gl[np.ix_(leadR_idx_whole,leadR_idx_whole)]),Sigma[np.ix_(leadR_idx_whole,sys_idx_whole)])


        #IpGRSigmaR = Isys+np.matmul(GR[np.ix_(sys_idx_whole,leadL_idx_whole)],Sigma[np.ix_(leadL_idx_whole,sys_idx_whole)])+np.matmul(GR[np.ix_(sys_idx_whole,leadR_idx_whole)],Sigma[np.ix_(leadR_idx_whole,sys_idx_whole)])
        #IpSigmaAGA = Isys+np.matmul(Sigma[np.ix_(sys_idx_whole,leadL_idx_whole)],GA[np.ix_(leadL_idx_whole,sys_idx_whole)])+np.matmul(Sigma[np.ix_(sys_idx_whole,leadR_idx_whole)],GA[np.ix_(leadR_idx_whole,sys_idx_whole)])
        #GlSS_glsys = np.matmul(np.matmul(IpGRSigmaR,gl[np.ix_(sys_idx_whole,sys_idx_whole)]),IpSigmaAGA)

        #



        #GRSS = GR[np.ix_(sys_idx_whole,sys_idx_whole)]
        #GASS = GA[np.ix_(sys_idx_whole,sys_idx_whole)]
        #GlSS1 = np.matmul(np.matmul(GRSS,SigmalSS),GASS)
        #GlSS1 += GlSS_glsys
        #GlSS2 = Gl[np.ix_(sys_idx_whole,sys_idx_whole)]

        #GlSSodd2 = GlSS1[np.ix_(odd_idx_sys,odd_idx_sys)]
        #GlSSeven2 = GlSS1[np.ix_(even_idx_sys,even_idx_sys)]

        #GRSSodd2 = np.linalg.solve(Isystemodd-np.matmul(gRSSodd,SigmaRSSodd1),gRSSodd)
        #GRSSeven2 = np.linalg.solve(Isystemeven-np.matmul(gRSSeven,SigmaRSSeven1),gRSSeven)


        #GlSSodd2 = np.matmul(np.matmul(GRSSodd1,SigmalSSodd1),GASSodd1)
        #GlSSodd2 += np.matmul(np.matmul((Isystemodd+np.matmul(GRSSodd2,SigmaRSSodd1)),glSSodd),(Isystemodd+np.matmul(SigmaRSSodd1.conj().T,GRSSodd2.conj().T)))

        #GlSSeven2 = np.matmul(np.matmul(GRSSeven1,SigmalSSeven1),GASSeven1)
        #GlSSeven2 += np.matmul(np.matmul((Isystemeven+np.matmul(GRSSeven2,SigmaRSSeven1)),glSSeven),(Isystemeven+np.matmul(SigmaRSSeven1.conj().T,GRSSeven2.conj().T)))

        # Way 2 to calculate current
        # Current matrices
        #SigmaRGlSSodd = np.matmul(SigmaRSSLodd1[np.ix_(LC_idx_odd_sys,LC_idx_odd_sys)],GlSSodd1[np.ix_(LC_idx_odd_sys,LC_idx_odd_sys)])
        #SigmalGASSodd = np.matmul(SigmalSSLodd1[np.ix_(LC_idx_odd_sys,LC_idx_odd_sys)],GASSodd1[np.ix_(LC_idx_odd_sys,LC_idx_odd_sys)])
        #GRSigmalSSodd = np.matmul(GRSSodd1[np.ix_(LC_idx_odd_sys,LC_idx_odd_sys)],SigmalSSLodd1[np.ix_(LC_idx_odd_sys,LC_idx_odd_sys)])
        #GlSigmaASSodd = np.matmul(GlSSodd1[np.ix_(LC_idx_odd_sys,LC_idx_odd_sys)],SigmaASSLodd1[np.ix_(LC_idx_odd_sys,LC_idx_odd_sys)])

        #SigmaRGlSSeven = np.matmul(SigmaRSSLeven1[np.ix_(LC_idx_even_sys,LC_idx_even_sys)],GlSSeven1[np.ix_(LC_idx_even_sys,LC_idx_even_sys)])
        #SigmalGASSeven = np.matmul(SigmalSSLeven1[np.ix_(LC_idx_even_sys,LC_idx_even_sys)],GASSeven1[np.ix_(LC_idx_even_sys,LC_idx_even_sys)])
        #GRSigmalSSeven = np.matmul(GRSSeven1[np.ix_(LC_idx_even_sys,LC_idx_even_sys)],SigmalSSLeven1[np.ix_(LC_idx_even_sys,LC_idx_even_sys)])
        #GlSigmaASSeven = np.matmul(GlSSeven1[np.ix_(LC_idx_even_sys,LC_idx_even_sys)],SigmaASSLeven1[np.ix_(LC_idx_even_sys,LC_idx_even_sys)])
        #print(LC_idx_even_sys)

        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #ax1.matshow(np.real(sigmaz_nn),interpolation='nearest')
        #ax1.set_xticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
        #ax1.set_yticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
        #ax1.set_title("sigmaz_nn")
        #ax1.grid()



#        cur_matrix_odd = np.matmul(sigmaz_nn,SigmaRGlSSodd+SigmalGASSodd-GRSigmalSSodd-GlSigmaASSodd)
#        cur_matrix_even = np.matmul(sigmaz_nn,SigmaRGlSSeven+SigmalGASSeven-GRSigmalSSeven-GlSigmaASSeven)
#        fig = plt.figure()
#        ax1 = fig.add_subplot(221)
#        ax1.matshow(np.abs(cur_matrix_odd),interpolation='nearest')
#        ax1.set_xticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
#        ax1.set_yticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
#        ax1.set_title("cur_matrix_odd")
#        ax1.grid()
#
#        ax2 = fig.add_subplot(222)
#        ax2.matshow(np.abs(cur_matrix_even),interpolation='nearest')
#        ax2.set_xticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
#        ax2.set_yticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
#        ax2.set_title("cur_matrix_even")
#        ax2.grid()
#
#        ax3 = fig.add_subplot(223)
#        ax3.matshow(np.abs(cur_matrix_odd1),interpolation='nearest')
#        ax3.set_xticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
#        ax3.set_yticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
#        ax3.set_title("cur_matrix_odd1")
#        ax3.grid()
#
#        ax4 = fig.add_subplot(224)
#        ax4.matshow(np.abs(cur_matrix_even1),interpolation='nearest')
#        ax4.set_xticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
#        ax4.set_yticks(np.arange(1.5,len(sigmaz_nn)-1.5,2))
#        ax4.set_title("cur_matrix_even1")
#        ax4.grid()
#
#        plt.show()
#


        #Is2[j] += 2.0*np.real(np.trace(cur_matrix_odd+cur_matrix_even))

    Is1[j] *= (V/2)/(n_Es-1)*1.0/(2*np.pi)
    #Is2[j] *= V/(n_Es-1)*1.0/(2*np.pi)

    #print(Is1[j]," ",Is2[j])

    #stop = timeit.default_timer()
    #print('Time: ',stop-start)
    #ISL *= V/(n_Es-1)*1.0/(2*np.pi)
    #for l in range(2*n_cutoff+1):
    #    print(str(l-n_cutoff)+" "+str(ISL[l]))

    #n_ts = 1000
    #It = np.zeros(n_ts,dtype=complex)
    #ts = np.linspace(0,10,n_ts)
    #for i in range(n_ts):
    #    t = ts[i]
    #    for j in range(2*n_cutoff+1):
    #        It[i] += np.exp(1.0j*(j-n_cutoff)*V*t)*ISL[j]
    #        #nt[i] += np.exp(1.0j*(j-n_cutoff)*V*t)*ns[j]
    
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
ax1.set_title("Is1")

ax2 = fig.add_subplot(122)
ax2.plot(Vs/DeltaL,np.real(Is2)/DeltaL)
ax2.set_xlabel("Voltage/$\Delta$")
ax2.set_ylabel("Current/$\Delta$")
ax2.set_title("Is2")



plt.show()
