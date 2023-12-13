import numpy as np
from matplotlib import pyplot as plt

from numpy import linalg as LA
from matplotlib.widgets import Slider





def fd_dist(E,T):
    if( T < 1e-5 ):
        if( E < 0 ):
            return 1.0
        else:
            return 0.0
    else:
        return 1.0/(np.exp(E/T)+1.0)

def tanh_T(E,T):
    if( T < 1e-5):
        if(E < 0):
            return -1.0
        else:
            return 1.0
    else:
        return np.tanh(E/(2*T))


def HBdG_k(k,model_pars,Hartree,Delta):
    epsA = model_pars["epsA"]
    epsB = model_pars["epsB"]
    tAA = model_pars["tAA"]
    tAB = model_pars["tAB"]
    U = model_pars["U"]
    T = model_pars["T"]
    a = model_pars["a"]

    HBdG_k = np.zeros([4,4],dtype=complex)
    HBdG_k[0,0] = epsA + Hartree[0] + 2*np.real(tAA*np.exp(1.0j*k*a))
    HBdG_k[1,1] = -epsA - Hartree[0] - 2*np.real(tAA*np.exp(1.0j*k*a))
    HBdG_k[2,2] = epsB + Hartree[1]
    HBdG_k[3,3] = -epsB - Hartree[1]

    HBdG_k[2,0] = np.conj(tAB)+tAB*np.exp(-1.0j*k*a)
    HBdG_k[3,1] = -(tAB+np.conj(tAB)*np.exp(1.0j*k*a))
    HBdG_k[0,2] = tAB+np.conj(tAB)*np.exp(1.0j*k*a)
    HBdG_k[1,3] = -(np.conj(tAB)+tAB*np.exp(-1.0j*k*a))

    HBdG_k[0,1] = Delta[0]
    HBdG_k[1,0] = np.conj(Delta[0])
    HBdG_k[2,3] = Delta[1]
    HBdG_k[3,2] = np.conj(Delta[1])
    return HBdG_k

def update_scf(n_ks,model_pars,Hartree0,Delta0):
    a = model_pars["a"]
    ks = np.zeros([n_ks])
    for n in range(n_ks):
        ks[n] = -np.pi/a + 2.0*np.pi*n/(a*n_ks)
    n_sites = Hartree0.size
    Hartree = np.zeros(Hartree0.size,dtype=complex)
    Delta = np.zeros(Hartree0.size,dtype=complex)
    fd_vec = np.vectorize(fd_dist)
    tanh_T_vec = np.vectorize(tanh_T)
    for n in range(n_ks):
        Es,Vs = LA.eigh(HBdG_k(ks[n],model_pars,Hartree0,Delta0))
        Es_fd = fd_vec(Es,T)
        Es_fd_diag = np.diag(Es_fd)
        VsEs_fdVs = np.matmul(Vs,np.matmul(Es_fd_diag,Vs.conj().T))
        for j in range(n_sites):
            Hartree[j] += U*VsEs_fdVs[2*j,2*j]
            Delta[j] += U*VsEs_fdVs[2*j,2*j+1]
        #us = Vs[0::2,n_sites::]
        #vs = Vs[1::2,n_sites::]

        #fd_Esp = fd_vec(Es[n_sites:],T)
        #fd_Esm = fd_vec(-Es[n_sites:],T)
        #tanh_Es = tanh_T_vec(Es[n_sites:],T)
        #for m in range(n_sites):
        #    for j in range(n_sites):
        #        Hartree[j] += U*(us[j,m]*np.conj(us[j,m])*fd_Esp[m]+vs[j,m]*np.conj(vs[j,m])*fd_Esm[m])
        #        Delta[j] += -U*us[j,m]*np.conj(vs[j,m])*tanh_Es[m]
        
        #Hartree1 = U*np.diag(np.matmul(us,np.matmul(np.diag(fd_Esp),us)))
        #Hartree2 = U*np.diag(np.matmul(vs,np.matmul(np.diag(fd_Esm),vs)))
        #Hartree += Hartree1+Hartree2
        #Delta += -U*np.diag(np.matmul(us,np.matmul(np.diag(tanh_Es),vs)))
    Hartree *= 1/n_ks
    Delta *= 1/n_ks

    return Hartree, Delta0

def scf_iteration(n_ks,model_pars,Hartree0,Delta0,tol):
    alpha = 1.0
    iteration = 0
    Hartree = Hartree0
    Delta = Delta0
    err = 1.0
    while( err > tol ):
        iteration += 1
        Hartree_new,Delta_new = update_scf(n_ks,model_pars,Hartree,Delta)
        err = max(np.amax(np.abs(Hartree_new-Hartree)),np.amax(np.abs(Delta_new-Delta)))
        if(iteration%5 == 0):
            print(iteration)
            print(err)
        if( err < tol):
            Hartree = Hartree_new
            Delta = Delta_new
            break
        else:
            Hartree += alpha*(Hartree_new-Hartree)
            Delta += alpha*(Delta_new-Delta)
    return Hartree,Delta


a = 1.0

T = 0.0

epsA = 0.0
epsB = 0.0
#epsA = -3.5
#epsB = -3.5

tAA = -1.0
tAB = np.sqrt(2)*tAA

U = -1.0


model_pars = {
    "epsA": epsA,
    "epsB": epsB,
    "tAA": tAA,
    "tAB": tAB,
    "U": U,
    "T": T,
    "a": a
}

tol = 1.0e-5


n_ks = 300
#ks = np.linspace(0,2*np.pi/2,n_ks,endpoint=False)
ks = np.zeros([n_ks])
for n in range(n_ks):
    ks[n] = -np.pi/a + 2.0*np.pi*n/(a*n_ks)


if True:

    n_gates = 100
    #gates = np.linspace(-4.5,2.5,n_gates)
    gates = np.linspace(-4.5,2.5,n_gates)
    n_sites = 2
    
    E_ks = np.zeros([n_gates,n_ks,4])
    
    pnums_tot = np.zeros(n_gates)
    pnums = np.zeros([n_gates,2])
    pairs = np.zeros([n_gates,2])

    mass_invs = np.zeros([4,n_gates])
    masss = np.zeros([4,n_gates])

    fd_vec = np.vectorize(fd_dist)
    for n in range(n_gates):
        model_pars["epsA"] = epsA-gates[n]
        model_pars["epsB"] = epsB-gates[n]
        #Hartree0 = np.array([-0.61,-0.49],dtype=complex)
        #Delta0 = np.array([0.3,0.22],dtype=complex)
        Hartree = np.array([-0.61,-0.49],dtype=complex)
        Delta = np.array([0.3,0.22],dtype=complex)
        #Hartree0 = np.array([-0.6,-0.4],dtype=complex)
        #Delta0 = np.array([1.0,1.0],dtype=complex)
        #Hartree,Delta = scf_iteration(n_ks,model_pars,Hartree0,Delta0,tol)
    
        
        
        
        

        for i in range(n_ks):
            E,V = LA.eigh(HBdG_k(ks[i],model_pars,Hartree,Delta))
            E_ks[n,i,:] = E
            Es_fd = fd_vec(E,T)
            Es_fd_diag = np.diag(Es_fd)
            VsEs_fdVs = np.matmul(V,np.matmul(Es_fd_diag,V.conj().T))
            for j in range(n_sites):
                pnums[n,j] += np.abs(VsEs_fdVs[2*j,2*j])
                pairs[n,j] += np.abs(VsEs_fdVs[2*j,2*j+1])

        zero_idx = int(n_ks/2)
        mass_invs[:,n] = (E_ks[n,zero_idx+1,:]+E_ks[n,zero_idx-1,:]-2*E_ks[n,zero_idx,:])/(ks[1]-ks[0])**2
        #masss = np.reciprocal(mass_invs)


        pnums_tot[n] += np.sum(pnums[n,:])
    
    pnums /= n_ks
    pairs /= n_ks
    pnums_tot /= n_ks
    
    fig1,ax1 = plt.subplots()
    ax1.plot(gates,pnums_tot)
    
    fig1,axs1 = plt.subplots(1,2)
    axs1[0].plot(gates,pnums[:,0])
    axs1[1].plot(gates,pnums[:,1])
    
    fig1,axs1 = plt.subplots(1,2)
    axs1[0].plot(gates,pairs[:,0])
    axs1[1].plot(gates,pairs[:,1])
    
    
    def band_structure(n):
        return E_ks[n,:,:]
    
    fig3,ax3 = plt.subplots()
    ax3.set_xlim([ks[0]-0.2,ks[-1]+0.2])
    ylim_max = np.amax(E_ks)+0.5
    ylim_min = np.amin(E_ks)-0.5
    ax3.set_ylim([ylim_min,ylim_max])
    line1, = ax3.plot(ks,band_structure(0)[:,0])
    line2, = ax3.plot(ks,band_structure(0)[:,1])
    line3, = ax3.plot(ks,band_structure(0)[:,2])
    line4, = ax3.plot(ks,band_structure(0)[:,3])

    
    ax3_pos = ax3.get_position()
    ax_gate_slider = fig3.add_axes([ax3_pos.x0,0.03,ax3_pos.width,0.03])
    gate_slider = Slider(
        ax = ax_gate_slider,
        label = "gate",
        valmin = gates[0],
        valmax = gates[-1],
        valstep = gates[1]-gates[0],
        valinit = gates[0]
    )
    
    def slider_update(val):
        n = int((val-gates[0])/(gates[1]-gates[0]))
        line1.set_ydata(band_structure(n)[:,0])
        line2.set_ydata(band_structure(n)[:,1])
        line3.set_ydata(band_structure(n)[:,2])
        line4.set_ydata(band_structure(n)[:,3])
        fig3.canvas.draw_idle()
    
    gate_slider.on_changed(slider_update)
    fig4, ax4 = plt.subplots()
    ax4.plot(gates,mass_invs[0,:])
    ax4.plot(gates,mass_invs[1,:])
    ax4.plot(gates,mass_invs[2,:])
    ax4.plot(gates,mass_invs[3,:])

#Hartree = np.array([-0.61,-0.49],dtype=complex)
#Delta = np.array([0.3,0.22],dtype=complex)
#
#model_pars["epsA"] = -2 - np.real(Hartree[0]+Hartree[1])/2
#model_pars["epsB"] = -2 - np.real(Hartree[0]+Hartree[1])/2
#E_ks = np.zeros([4,n_ks])
#for i in range(n_ks):
#    E,V = LA.eigh(HBdG_k(ks[i],model_pars,Hartree,Delta))
#    E_ks[:,i] = E






# Animation of the filling of the system
#frames = []
#fig = plt.figure()
#for n in range(n_gates):
#    frames.append([plt.plot(E_ks[n,


#err = 1.0
#print("solution with error ", err, " at iteration ", iteration)
#print("Hartree = ", Hartree, " Delta = ",Delta)
#print("Hartree sum = ", np.sum(Hartree))




#fig = plt.figure()
#plt.plot(ks,E_ks[0,:])
#plt.plot(ks,E_ks[1,:])
#plt.plot(ks,E_ks[2,:])
#plt.plot(ks,E_ks[3,:])

zero_idx = int(n_ks/2)
print(ks[zero_idx-1],ks[zero_idx],ks[zero_idx+1])
inv_eff_mass = (E_ks[:,zero_idx+1]+E_ks[:,zero_idx-1]-2*E_ks[:,zero_idx])/(ks[1]-ks[0])**2
eff_mass = 1.0/inv_eff_mass
print("eff_masses", eff_mass)



plt.show()

