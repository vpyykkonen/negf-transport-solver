import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


#np.set_printoptions(threshold=sys.maxsize)

def fd(E,T):
    if np.abs(T) < 1e-5:
        if E > 0.0:
            return 0.0
        else:
            return 1.0

    return 1.0/(1.0+np.exp(E/T))

eQD = -0.0
tSL = np.sqrt(30)
tSR = np.sqrt(30)
tLS = tSL
tRS = tSR

muL = 0.0
muR = 0.0
tL = 30
tR = 30
DeltaL = 1.0
DeltaR = 1.0
ieta = 1.0j*1.0e-3
T = 0

cutoff_energy = 20;


n_Es = 200
n_Vs = 50
Vs = np.linspace(0.25,1.5,n_Vs)
Is = np.zeros(n_Vs)

sigmaz = np.matrix([[1.0,0.0],[0.0,-1.0]],dtype=complex)

for j in range(n_Vs):
    V = Vs[j]
    n_cutoff = math.ceil(cutoff_energy/V)
    I = np.eye(2*(2*n_cutoff+1),dtype = complex)
    sigmaz_nn = np.kron(np.eye(2*n_cutoff+1,dtype = complex),sigmaz)
    Es = np.linspace(0,abs(V),n_Es)
    print("V = "+str(V))
    for i in range(n_Es):
        E = Es[i]
        SigmaR0 = np.asmatrix(np.zeros((2*(2*n_cutoff+1),2*(2*n_cutoff+1)), dtype = complex))
        Sigmal0 = np.asmatrix(np.zeros((2*(2*n_cutoff+1),2*(2*n_cutoff+1)), dtype = complex))
        gR = np.asmatrix(np.zeros((2*(2*n_cutoff+1),2*(2*n_cutoff+1)), dtype = complex))
        gA = np.asmatrix(np.zeros((2*(2*n_cutoff+1),2*(2*n_cutoff+1)), dtype = complex))
        gl = np.asmatrix(np.zeros((2*(2*n_cutoff+1),2*(2*n_cutoff+1)), dtype = complex))
        for k in range(-n_cutoff,n_cutoff+1,1):
            # Definition of the lead self-energy
            # self-energy couples energies V apart due to the 
            # pair Green's functions in the leads
    
            #gRLL = np.array([[-(E+ieta),-DeltaL],[-DeltaL,-(E+ieta)]])
            #gRLL *= 1.0/tL*1/sqrt((E+ieta)**2-abs(DeltaL)**2)
    
            #gRRR = np.array([[-(E+ieta),-DeltaR],[-DeltaR,-(E+ieta)]])
            #gRRR *= 1.0/tR*1/sqrt((E+ieta)**2-abs(DeltaR)**2)
    
            # Retarded 
            gRLLp_const = 1.0/(tL*np.sqrt(np.abs(DeltaL)**2-(E+k*V+V/2+ieta)**2))
            gRLLm_const = 1.0/(tL*np.sqrt(np.abs(DeltaL)**2-(E+k*V-V/2+ieta)**2))
                                                           
            gRRRp_const = 1.0/(tR*np.sqrt(np.abs(DeltaR)**2-(E+k*V+V/2+ieta)**2))
            gRRRm_const = 1.0/(tR*np.sqrt(np.abs(DeltaR)**2-(E+k*V-V/2+ieta)**2))

            gRLLp = gRLLp_const*np.matrix([[-(E+k*V+V/2+ieta),DeltaL],[np.conj(DeltaL),-(E+k*V+V/2+ieta)]],dtype = complex)
            gRLLm = gRLLm_const*np.matrix([[-(E+k*V-V/2+ieta),DeltaL],[np.conj(DeltaL),-(E+k*V-V/2+ieta)]],dtype = complex)
            gRRRp = gRRRp_const*np.matrix([[-(E+k*V+V/2+ieta),DeltaR],[np.conj(DeltaR),-(E+k*V+V/2+ieta)]],dtype = complex)
            gRRRm = gRRRm_const*np.matrix([[-(E+k*V-V/2+ieta),DeltaR],[np.conj(DeltaR),-(E+k*V-V/2+ieta)]],dtype = complex)

            gALLp = gRLLp.getH()
            gALLm = gRLLm.getH()
            gARRp = gRRRp.getH()
            gARRm = gRRRm.getH()

            glLLp = (gALLp-gRLLp)*fd(E+k*V+V/2,T)
            glLLm = (gALLm-gRLLm)*fd(E+k*V-V/2,T)
            glRRp = (gARRp-gRRRp)*fd(E+k*V+V/2,T)
            glRRm = (gARRm-gRRRm)*fd(E+k*V-V/2,T)

            n = k+n_cutoff # matrix index
            #print(str(n) + " " + str(k))
            #print(gRLL11p)
            #print(gRRR11m)

            SigmaR0[2*n,2*n] = tSL*tLS*gRLLp[0,0] + tSR*tRS*gRRRm[0,0]
            Sigmal0[2*n,2*n] = tSL*tLS*glLLp[0,0] + tSR*tRS*glRRm[0,0]

            SigmaR0[2*n+1,2*n+1] = tSL*tLS*gRLLm[1,1] + tSR*tRS*gRRRp[1,1]
            Sigmal0[2*n+1,2*n+1] = tSL*tLS*glLLm[1,1] + tSR*tRS*glRRp[1,1]
    
            if k > -n_cutoff:
                SigmaR0[2*n,2*(n-1)+1] =-tSR*tRS*gRRRm[0,1]
                SigmaR0[2*n+1,2*(n-1)] =-tSL*tLS*gRLLm[1,0]
                Sigmal0[2*n,2*(n-1)+1] =-tSR*tRS*glRRm[0,1]
                Sigmal0[2*n+1,2*(n-1)] =-tLS*tLS*glLLm[1,0]
    
            if k < n_cutoff:
                SigmaR0[2*n,2*(n+1)+1] =-tSL*tSL*gRLLp[0,1]
                SigmaR0[2*n+1,2*(n+1)] =-tSR*tSR*gRRRp[1,0]
                Sigmal0[2*n,2*(n+1)+1] =-tSL*tSL*glLLp[0,1]
                Sigmal0[2*n+1,2*(n+1)] =-tSR*tSR*glRRp[1,0]
    
            gRnn = np.linalg.inv(np.matrix([[E+k*V+ieta-eQD,0.0],[0.0,E+k*V+ieta+eQD]]))
            gAnn = gRnn.getH()
            glnn = (gAnn-gRnn)*fd(E+k*V,T)
            
    
            gR[2*n:2*n+2,2*n:2*n+2] = gRnn
            gA[2*n:2*n+2,2*n:2*n+2] = gAnn
            gl[2*n:2*n+2,2*n:2*n+2] = glnn
    
        GR = np.linalg.solve(I-np.matmul(gR,SigmaR0),gR)
        #Gl = np.matmul(np.matmul((I+np.matmul(GR,SigmaR0)),gl),(I+np.matmul(SigmaR0.conj().T,GR.conj().T)))+ np.matmul(np.matmul(GR,Sigmal0),GR.conj().T)
        Gl = np.matmul(np.matmul(GR,Sigmal0),GR.getH())
        #print(np.linalg.norm(GR.conj().T))
        #print(np.linalg.norm(GR))
        #print(np.linalg.norm(Sigmal0))
        #print(np.linalg.norm(Gl))
       #print(Gl)

        cur_matrix = np.matmul(sigmaz_nn,np.matmul(SigmaR0,Gl)+np.matmul(Sigmal0,GR.getH())-np.matmul(GR,Sigmal0)-np.matmul(Gl,SigmaR0.getH()))
        Is[j] += np.real(np.trace(cur_matrix))
        #Is[j] += -2*np.trace(cur_matrix)
    Is[j] *= V/(n_Es-1)*1.0/(2*np.pi)
    print(Is[j])
        #for k in range(-n_cutoff,n_cutoff+1,1):
        #    n = k+n_cutoff # matrix index
        #    ISL[n] += -2*np.real(np.trace(cur_matrix,offset=2*k))
            #ns[n] += -1.0j*np.trace(Gl[::2,::2],offset=k)

    #ISL *= V/(n_Es-1)*1/(4*np.pi**2)
    #Is[j] = ISL[n_cutoff]

        
#    ns *= V/(n_Es-1)*1/(4*np.pi**2)
#    for i in range(2*n_cutoff+1):
#        print(str(-n_cutoff+i)+" "+str(ISL[i]))
#    for i in range(2*n_cutoff+1):
#        print(str(-n_cutoff+i)+" "+str(ns[i]))
#
#n_ts = 1000
#ts = np.linspace(0,10,n_ts)
#It = np.zeros(n_ts,dtype=complex)
#nt = np.zeros(n_ts,dtype=complex)
#for i in range(n_ts):
#    t = ts[i]
#    for j in range(2*n_cutoff+1):
#        It[i] += np.exp(1.0j*(j-n_cutoff)*V*t)*ISL[j]
#        nt[i] += np.exp(1.0j*(j-n_cutoff)*V*t)*ns[j]
#
#fix, ax1 = plt.subplots()
#ax1.plot(ts,It)
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



fig,ax1 =  plt.subplots()
ax1.plot(Vs,np.real(Is))


plt.show()





            










