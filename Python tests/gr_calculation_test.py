import numpy as np
import matplotlib.pyplot as plt
import sys

eQD = 0.0
tLR = 1.0
tRL = 1.0
ieta = 0.01*1.0j
DeltaL = 1.0
DeltaR = 1.0
tL = 20.0

n_Es = 100
Es = np.linspace(-5,5,n_Es)

Sigma = np.zeros([4,4], dtype=complex)

HLR = np.matrix([[tLR,0.0],[0.0,-tRL]],dtype = complex)
HRL = np.matrix([[tRL,0.0],[0.0,-tLR]],dtype = complex)
print(HLR)
print(HRL)

Sigma[0,2] = tLR
Sigma[1,3] = -tRL
Sigma[2,0] = tRL
Sigma[3,1] = -tLR
print(Sigma)

gR = np.zeros([4,4],dtype=complex)

I4 = np.eye(4,dtype=complex)
I2 = np.eye(2,dtype=complex)


for i in range(n_Es):
    E = Es[i]
    gRLL_const = 1.0/(tL*np.sqrt(np.abs(DeltaL)**2-(E+ieta)**2))
    gRRR_const = 1.0/(tL*np.sqrt(np.abs(DeltaR)**2-(E+ieta)**2))
    gRLL = gRLL_const*np.matrix([[-(E+ieta),DeltaL],[DeltaL,-(E+ieta)]],dtype=complex)
    gRRR = gRRR_const*np.matrix([[-(E+ieta),DeltaR],[DeltaR,-(E+ieta)]],dtype=complex)
    gR[0:2,0:2] = gRLL
    gR[2:4,2:4] = gRRR
    #print("gR = ")
    #print(gR)
    #print(gR.shape)
    #print(gR*Sigma)

    GR1 = np.linalg.solve(I4-np.matmul(gR,Sigma),gR)

    SigmaL = np.matmul(HRL,np.matmul(gRLL,HLR))
    SigmaR = np.matmul(HLR,np.matmul(gRRR,HRL))

    GRLL = np.linalg.solve(I2-np.matmul(gRLL,SigmaR),gRLL)
    GRRR = np.linalg.solve(I2-np.matmul(gRRR,SigmaL),gRRR)

    GRLR = np.matmul(np.matmul(gRLL,HLR),GRRR)
    GRRL = np.matmul(np.matmul(gRRR,HRL),GRLL)

    GR2 = np.block([[GRLL,GRLR],[GRRL,GRRR]])

    #print("GR1 =\n")
    #print(GR1)
    #print("GR2 =\n")
    #print(GR2)
    
    #print(GR1.shape)
    #print(GR2.shape)


    print("Difference between methods")
    print(np.linalg.norm(GR2-GR1))





    


        
