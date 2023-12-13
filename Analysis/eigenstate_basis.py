# A tool to analyze scattering system eigenlevels and their connection
# to the leads

import sys
import re
import numpy as np
import h5py
from os import path
import csv
import os

import networkx as nx
import matplotlib.pyplot as plt

from numpy import linalg as LA

pair = re.compile(r'\(([^,\)]+),([^,\)]+)\)')
def parse_cppcomplex(s):
    return complex(*map(float, pair.match(s.strip()).groups()))

def read_complex_matrix(file_name):
    f = open(file_name,"r")
    Rows = f.readlines()
    f.close()
    data = []

    for row in Rows:
        row_elems = list(map(parse_cppcomplex,row.split(";")))
        data.append(row_elems)


    return np.array(data,dtype=complex)

def read_matrix(file_name):
    f = open(file_name,"r")
    Rows = f.readlines()
    f.close()
    data = []

    for row in Rows:
        row_elems = list(map(string_to_num,row.split(";")))
        data.append(row_elems)

    return np.array(data,dtype = complex)

def show_graph_with_labels(adjacency_matrix, mylabels):
    fig = plt.figure()
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    #nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    nx.draw(gr, node_size=500)
    #plt.show()

def get_array_from_datapoint(datapoint_path,data_path):
    if datapoint_path[-3:] == '.h5':
        f = h5py.File(datapoint_path,'r')
        array_r = np.matrix(f.get(data_path+'_r')).astype(np.complex128)
        array_i = np.matrix(f.get(data_path+'_i')).astype(np.complex128)
        array = np.zeros(array_r.shape,dtype=complex)
        array = array_r + 1.0j*array_i
        f.close()
        return array
    else:
        return read_complex_matrix(datapoint_path+'/'+data_path)


def load_config_file_to_dict(file_path):
    a  = {}
    with open(file_path,"r") as f:
        for line in f:
            line = line.replace("\n","")
            line = line.replace(" ","") # remove white space
            line = line.split('#',1)[0] # remove comments
            if line == "": # if line is empty, move to next
                continue
            key,value = line.split('=',1)
            a[key] = value

    return a




def geometry_parser(datapoint_path):
    a = {}
    if datapoint_path[-3:] == '.h5':
        f = h5py.File(datapoint_path,'r')
        a = dict(f.attrs.items())
        f.close()
    else:
        a = load_config_file_to_dict(datapoint_path+'/geometry.cfg')

    dim = int(a["dim"])
    #lattice_name = a["lattice_name"]
    unitcell_sites = int(a["unitcell_sites"])
    n_unitcells = list(map(int,a["n_unitcells"].split(',')))

    on_site = list(map(float,a["on_site"].split(',')))

    total_unitcells = 1
    dim_increments = [1]
    for n in range(dim):
        total_unitcells *= n_unitcells[n]
        if n > 0:
            dim_increments.append(dim_increments[n-1]*n_unitcells[n-1])



    n_hoppings = int(a["n_hoppings"])
    target_unitcells = []
    target_orbitals = []
    source_orbitals = []
    amplitudes = []

    for n in range(n_hoppings):
        hop_str = a["hopping"+str(n+1)]
        target_source_str,amp_str = hop_str.split('=',1)
        amp_str = amp_str.strip('(')
        amp_str = amp_str.strip(')')
        amp_r,amp_i = list(map(float,amp_str.split(',',1)))
        amplitudes.append(amp_r + 1.0j*amp_i)

        target_unitcell_str,source_orbital_str,target_orbital_str = target_source_str.split(';')
        target_unitcells.append( list(map(int,target_unitcell_str.split(','))))
        source_orbitals.append(int(source_orbital_str))
        target_orbitals.append(int(target_orbital_str))


    edge_strs = a["edge"].split(';')
    remove_orbs = []
    for edge_str in edge_strs:
        orbs_list = []
        remove_orbs_str = edge_str.split(',')
        for remove_orb_str in remove_orbs_str:
            if remove_orb_str:
                orbs_list.append(int(remove_orb_str))
        remove_orbs.append(orbs_list)

    n_sites = total_unitcells * unitcell_sites
    

    H0 = np.zeros([n_sites,n_sites],dtype=complex)
    for n in range(total_unitcells):
        target_idx = 0
        source_idx = 0
        for m in range(unitcell_sites):
            target_idx = unitcell_sites*n + m
            source_idx = unitcell_sites*n + m
            H0[target_idx,source_idx] = on_site[m]
        for i in range(n_hoppings):
            source_idx = unitcell_sites*n + source_orbitals[i]
            target_idx = unitcell_sites*n + target_orbitals[i]
            for d in range(dim):
                target_idx += unitcell_sites*target_unitcells[i][d]*dim_increments[d]

    
            # Add hopping in case that it does not go out of bounds
            if target_idx >= 0 and target_idx < n_sites:
                H0[target_idx,source_idx] = amplitudes[i]
                H0[source_idx,target_idx] = np.conj(amplitudes[i]) # hermitian conjugate

    remove_idxs = []
    edge_sizes = []
    edge_increments = []
    for d1 in range(dim):
        prod = 1
        edge_increment = [prod]
        for d2 in range(dim):
            if d1 != d2:
                prod *= n_unitcells[d2]
                edge_increment.append(prod)
        edge_sizes.append(prod)
        edge_increments.append(edge_increment)


    for d1 in range(dim):
        for orb in remove_orbs[2*d1]:
            for n in range(edge_sizes[d1]):
                idx = orb
                edge_dim = 0
                for d2 in range(dim):
                    if d2 != d1:
                        idx += int(((n%edge_increments[d1][edge_dim+1])/edge_increments[d1][edge_dim])*dim_increments[d2]*unitcell_sites)
                        edge_dim += 1
                remove_idxs.append(idx)

        for orb in remove_orbs[2*d1+1]:
            for n in range(edge_sizes[d1]):
                idx = (n_unitcells[d1]-1)*dim_increments[d1]*unitcell_sites + orb
                edge_dim = 0
                for d2 in range(dim):
                    if d2 != d1:
                        idx += int(((n%edge_increments[d1][edge_dim+1])/edge_increments[d1][edge_dim])*dim_increments[d2]*unitcell_sites)
                        edge_dim += 1
                remove_idxs.append(idx)
    remove_idxs = np.unique(np.array(remove_idxs)).tolist()

    H0 = np.delete(H0,remove_idxs,0)
    H0 = np.delete(H0,remove_idxs,1)

    return H0

def get_density_and_pair(datapoint_path):
    Gl_time = np.zeros((1,1),dtype=complex)
    if datapoint_path[-3:] == '.h5':
        Gl_time = get_array_from_datapoint(datapoint_path,"data/Gl0_time")
    else:
        Gl_time = get_array_from_datapoint(datapoint_path,"Data/Gl_time.csv")
    pair = -1.0j*np.diagonal(Gl_time,1)[0::2]
    density = -1.0j*np.diagonal(Gl_time)[0::2]

    return density,pair


def visualize_matrix(matrix):
    m_rows,m_cols = matrix.shape
    Nsites = int(m_cols/2)
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    pos = ax.matshow(np.real(matrix),interpolation='nearest')
    ax.set_xticks(np.arange(0.5,len(matrix)-0.5,1))
    ax.set_yticks(np.arange(0.5,len(matrix)-0.5,1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Real part")
    fig.colorbar(pos)
    ax.grid()
    
    
    for i in range(Nsites-1):
        ax.axvline(x = (i+1)*2-0.5,color='black')
        ax.axhline(y = (i+1)*2-0.5,color='black')
    
    ax = fig.add_subplot(122)
    pos = ax.matshow(np.imag(matrix),interpolation='nearest')
    ax.set_xticks(np.arange(0.5,len(matrix)-0.5,1))
    ax.set_yticks(np.arange(0.5,len(matrix)-0.5,1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Imaginary part")
    fig.colorbar(pos)
    ax.grid()
    
    for i in range(Nsites-1):
        ax.axvline(x = (i+1)*2-0.5,color='black')
        ax.axhline(y = (i+1)*2-0.5,color='black')
    
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    pos = ax.matshow(np.abs(matrix),interpolation='nearest')
    ax.set_xticks(np.arange(0.5,len(matrix)-0.5,1))
    ax.set_yticks(np.arange(0.5,len(matrix)-0.5,1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Absolute value")
    fig.colorbar(pos)
    ax.grid()
    
    
    for i in range(Nsites-1):
        ax.axvline(x = (i+1)*2-0.5,color='black')
        ax.axhline(y = (i+1)*2-0.5,color='black')
    
    ax = fig.add_subplot(122)
    pos = ax.matshow(np.angle(matrix),interpolation='nearest')
    ax.set_xticks(np.arange(0.5,len(matrix)-0.5,1))
    ax.set_yticks(np.arange(0.5,len(matrix)-0.5,1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Phase angle")
    fig.colorbar(pos)
    ax.grid()
    
    for i in range(Nsites-1):
        ax.axvline(x = (i+1)*2-0.5,color='black')
        ax.axhline(y = (i+1)*2-0.5,color='black')
    
    plt.tight_layout()
    #plt.show()




np.set_printoptions(edgeitems=30, linewidth=100000, precision = 3)
        #formatter=dict(float=lambda x: "%.3g" % x))
if len(sys.argv) == 1:
    sys.exit("Path to dataset was not given.")
data_point_path = sys.argv[1]
H0 = geometry_parser(data_point_path)
n_sites = H0.shape[0]

density,pair = get_density_and_pair(data_point_path)


a = {}
if data_point_path[-3:] == '.h5':
    f = h5py.File(data_point_path,'r')
    a = dict(f.attrs.items())
    f.close()
else:
    a = load_config_file_to_dict(data_point_path+'/parameters.cfg')


# Construct HBdG
U = float(a["U"])
VCL = float(a["VCL"])
VCR = float(a["VCR"])
cpoint_L_str = a["cpoint_L"]
cpoint_R_str = a["cpoint_R"]
cpoint_L = 0
cpoint_R = 0
if cpoint_L_str == "end":
    cpoint_L = n_sites-1
else:
    cpoint_L = int(cpoint_L_str)
if cpoint_R_str == "end":
    cpoint_R = n_sites-1
else:
    cpoint_R = int(cpoint_R_str)

print("cpoint_L", cpoint_L)
print("cpoint_R", cpoint_R)

gate = float(a["gate"])
print("gate",gate)
Delta = U*pair
#Delta = np.real(Delta)
#Delta = 0.5*np.exp(1.0j*np.linspace(0,0.9*np.pi,n_sites))
#print("Delta",Delta)
#0.2*np.ones(n_sites,dtype=complex)
Hartree = U*density
Hartree -= Hartree.mean()

print("Delta",Delta)
print("Hartree",Hartree)
if density.shape[0] > n_sites:
    Delta = np.delete(Delta,[0,-1])
    Hartree = np.delete(Hartree,[0,-1])


HBdG = np.zeros([2*n_sites,2*n_sites],dtype=complex)
HBdG_Hart = np.zeros([2*n_sites,2*n_sites],dtype=complex)
HBdG_Delta = np.zeros([2*n_sites,2*n_sites],dtype=complex)
for i in range(n_sites):
    for j in range(n_sites):
        HBdG[2*i,2*j] = H0[i,j]
        HBdG[2*i+1,2*j+1] = -H0[j,i]
        HBdG_Hart[2*i,2*j] = H0[i,j]
        HBdG_Hart[2*i+1,2*j+1] = -H0[j,i]
        HBdG_Delta[2*i,2*j] = H0[i,j]
        HBdG_Delta[2*i+1,2*j+1] = -H0[j,i]
    HBdG[2*i,2*i] += -gate + Hartree[i]
    HBdG[2*i+1,2*i+1] += gate-Hartree[i]
    HBdG[2*i,2*i+1] = Delta[i]
    HBdG[2*i+1,2*i] = np.conj(Delta[i])
    HBdG_Hart[2*i,2*i] += -gate + Hartree[i]
    HBdG_Hart[2*i+1,2*i+1] += gate-Hartree[i]
    HBdG_Delta[2*i,2*i] += -gate
    HBdG_Delta[2*i+1,2*i+1] += gate
    HBdG_Delta[2*i,2*i+1] = Delta[i]
    HBdG_Delta[2*i+1,2*i] = np.conj(Delta[i])
    # Add boundary potentials
    if i == cpoint_L:
        HBdG[2*i,2*i] += VCL+0.01
        HBdG[2*i+1,2*i+1] -= VCL+0.01
        HBdG_Hart[2*i,2*i] += VCL+0.01
        HBdG_Hart[2*i+1,2*i+1] -= VCL+0.01
        HBdG_Delta[2*i,2*i] += VCL+0.01
        HBdG_Delta[2*i+1,2*i+1] -= VCL+0.01
    if i == cpoint_R:
        HBdG[2*i,2*i] += VCR+0.01
        HBdG[2*i+1,2*i+1] -= VCR+0.01
        HBdG_Hart[2*i,2*i] += VCR+0.01
        HBdG_Hart[2*i+1,2*i+1] -= VCR+0.01
        HBdG_Delta[2*i,2*i] += VCR+0.01
        HBdG_Delta[2*i+1,2*i+1] -= VCR+0.01


# lead system hoppings

tLS = float(a["tLS"])
tRS = float(a["tRS"])
HLS = np.matrix([[tLS,0.0],[0.0,-np.conj(tLS)]],dtype=complex)
HRS = np.matrix([[tRS,0.0],[0.0,-np.conj(tLS)]],dtype=complex)


# diagonalize HBdG without mean-field parameters
H0_add_local = np.copy(H0)
for i in range(n_sites):
    H0_add_local[i,i] -= gate
    if i == cpoint_L:
        H0_add_local[i,i] += VCL
    if i == cpoint_R:
        H0_add_local[i,i] += VCR

print(H0)
print(H0_add_local)
Es0,V0 = LA.eigh(H0_add_local)
print("Es0",Es0)

print("eff_hop0",tLS*V0[cpoint_L,:])


# diagonalize HBdG
    

#print(np.abs(HBdG))

Es,V = LA.eigh(HBdG)
Es_Hart,V_Hart = LA.eigh(HBdG_Hart)
Es_Delta,V_Delta = LA.eigh(HBdG_Delta)

visualize_matrix(HBdG)

Lambda = np.diag(Es)
Lambda_Hart = np.diag(Es_Hart)
Lambda_Delta = np.diag(Es_Delta)


# eigen vector components for cpoint_L

V_L = V[2*cpoint_L:2*cpoint_L+2,:]
V_R = V[2*cpoint_R:2*cpoint_R+2,:]

V_L_Hart = V_Hart[2*cpoint_L:2*cpoint_L+2,:]
V_L_Delta = V_Delta[2*cpoint_L:2*cpoint_L+2,:]
V_R_Hart = V_Hart[2*cpoint_R:2*cpoint_R+2,:]
V_R_Delta = V_Delta[2*cpoint_R:2*cpoint_R+2,:]
print("Es",Es)
#print("V",V)
#visualize_matrix(Es)
#visualize_matrix(V)
#visualize_matrix(V_L)
#visualize_matrix(V_R)

visualize_matrix(V)

print("V_L",V_L)
print("V_R",V_R)
print("left_hops")
print(np.abs(np.matmul(HLS,V_L)))
print("right_hops")
print(np.abs(np.matmul(HRS,V_R)))

print("left_hops_Hart")
print(np.abs(np.matmul(HLS,V_L_Hart)))
print("right_hops_Hart")
print(np.abs(np.matmul(HRS,V_R_Hart)))

print("left_hops_Delta")
print(np.abs(np.matmul(HLS,V_L_Delta)))
print("right_hops_Delta")
print(np.abs(np.matmul(HRS,V_R_Delta)))

#H0_graph = (np.abs(H0) > 1e-12).astype(int)
#np.fill_diagonal(H0_graph,0)
#print(H0_graph)
#
#labels = list(range(H0.shape[0]))
#show_graph_with_labels(np.abs(H0_graph),labels)

plt.show()
    
    

















            

                














