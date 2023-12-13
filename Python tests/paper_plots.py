import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from os import path
import csv
import os
import h5py

import matplotlib

from os.path import exists

# Regular expression for complex number matching
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

def string_to_num(num_str):
    pair_match = pair.match(num_str.strip())
    if pair_match:
        return complex(*map(float, pair_match.groups()))
    else:
        return float(num_str.strip())


def read_matrix(file_name):
    f = open(file_name,"r")
    Rows = f.readlines()
    f.close()
    data = []

    for row in Rows:
        row_elems = list(map(string_to_num,row.split(";")))
        data.append(row_elems)

    return np.array(data,dtype = complex)

def parse_parameter_file(path):
    params = {}
    with open(path) as f:
        for line in f:
            re.sub(r'#.*','',line)
            if("=" in line):
                (key,val) = line.split("=",1)
                params[key.strip()] = val.strip()
    return params
        
if len(sys.argv) == 1:
    sys.exit("Specifications not given.")

if sys.argv[1] == "ss_ne":
    ss_ne_data_path = "../Data/SS_ne_Sawtooth4_gate100:-4.5:3.0_bias0.50_U-0.3_tLS5.3_VCL1.0_DeltaL1.00_DeltaR1.00_TL0"
    #ss_ne_data_path = "../Data/SS_ne_Sawtooth4_gate100:-4.5:3.0_bias0.50_U0_tLS5.3_VCL1.0_DeltaL1.00_DeltaR1.00_TL0"

    if len(sys.argv) > 2:
        ss_ne_data_path = sys.argv[2]
    
    
    data_points_list_file = open(ss_ne_data_path + '/data_points.csv')
    #
    csvreader = csv.reader(data_points_list_file)
    variables = []
    variables = next(csvreader)
    data_points_list_file.close()
    folders = ss_ne_data_path.split("/")
    folder_name = folders[-1]
    if folders[-1] == "":
        folder_name = folders[-2]
    
    data_points = genfromtxt(ss_ne_data_path +\
            '/data_points.csv',delimiter=';',skip_header=1)[:,1:]
    num_points,num_vars = data_points.shape
    
    params = parse_parameter_file(ss_ne_data_path+'/parameters_const.cfg')
    
    hdf5 = False
    if exists(ss_ne_data_path+"/1.h5"):
        hdf5 = True
    else:
        hdf5 = False
    
    U = float(params["U"])
    bias = float(params["bias"])
    gate = float(params["gate"])
    tLS = float(params["tLS"])
    tRS = float(params["tRS"])
    TL = float(params["TL"])
    TR = float(params["TR"])
    tL = float(params["tL"])
    tR = float(params["tR"])
    VCL = float(params["VCL"])
    VCR = float(params["VCR"])
    #ieta = parse_complex(params("ieta"))
    
    n_sites = 0
    n_harmonics = 0
    rows = n_harmonics
    cols = n_harmonics
    for i in range(num_points):
        if not hdf5:
            if path.exists(ss_ne_data_path + \
                    "/"+str(i+1)+"/Data/particle_number.csv"):
                rows,cols = read_matrix(ss_ne_data_path +\
                    "/"+str(i+1)+"/Data/particle_number.csv").shape
        else:
            f = h5py.File(ss_ne_data_path + "/"+str(i+1)+".h5","r")
            if "data/particle_number_r" in f:
                rows,cols = np.array(f.get("data/particle_number_r")).shape
            f.close()
        n_sites = max(n_sites,int(rows))
        n_harmonics = max(n_harmonics,int(cols-1))
    
    Gl_times = np.zeros((num_points,n_harmonics+1,2*(n_sites+2),2*(n_sites+2)),dtype=complex)
    curs = np.zeros((num_points,n_harmonics+1),dtype=complex)
    pnums = np.zeros((num_points,n_sites,n_harmonics+1),dtype=complex)
    pairs = np.zeros((num_points,n_sites,2*n_harmonics+1),dtype=complex)
    
    
    if n_harmonics == 0:
        Gl_times = np.zeros((num_points,1,2*(n_sites+2),2*(n_sites+2)),dtype=complex)
    
    not_found = []
    for i in range(num_points):
        if hdf5:
            f = h5py.File(ss_ne_data_path + "/"+str(i+1)+".h5","r")
            if "data/current_r" in f:
                # Load the lesser Green's function
                for n in range(n_harmonics+1):
                    Gl_path = "data/Gl"+str(n)+"_time"
                    Gl_mat = np.matrix(f.get(Gl_path+"_r")).astype(np.complex128)
                    Gl_mat += 1.0j*np.matrix(f.get(Gl_path+"_i")).astype(np.complex128)
                    Gl_times[i,n,:,:] = Gl_mat
    
                for n in range(n_harmonics+1):
                    if(n == 0):
                        curs[i,0] = -2.0*tLS*Gl_times[i,1,0,2]-2.0*tLS*np.conj(Gl_times[i,1,0,2])
                        #curs[i,0] = 2.0*tRS*np.conj(Gl_times[i,1,2*n_sites,2*(n_sites+1)])-2.0*tRS*Gl_times[i,1,2*n_sites,2*(n_sites+1)])
                    elif(n < n_harmonics):
                        curs[i,n] = -2.0*tLS*Gl_times[i,n+1,0,2] + 2.0*tLS*Gl_times[i,n-1,2,0]
                        #curs[i,0] = -2.0*tRS*Gl_times[i,n-1,2*n_sites,2*(n_sites+1)])+2.0*tRS*Gl_times[i,n+1,2*n_sites,2*(n_sites+1)])
                    
                    #curs[i,n] = -2.0*tLS*Gl_times[i,n
                    #curs[i,n] = -2.0*tAA*Gl_times[i,n,2,6] + 2.0*tAA*Gl_times[i,n,6,2]
                    #curs[i,n] += -2.0*tAB*Gl_times[i,n,4,6] + 2.0*tAB*Gl_times[i,n,6,4]
                    for j in range(n_sites):
                        pnums[i,j,n] = -2.0j*Gl_times[i,n,2*(j+1),2*(j+1)]
                        if(n == 0):
                            pairs[i,j,0] = -1.0j*Gl_times[i,0,2*(j+1),2*(j+1)+1]
                        else:
                            # Positive harmonic
                            pairs[i,j,2*n-1] = -1.0j*Gl_times[i,n,2*(j+1),2*(j+1)+1]
                            # Negative harmonic
                            pairs[i,j,2*n] = 1.0j*np.conj(Gl_times[i,n,2*(j+1)+1,2*(j+1)])
            else:
                not_found.append(i)
            f.close()
    
    print(not_found)
    
    fig, ax = plt.subplots()
    
    #hfont = {'fontname':'Helvetica'}
    
    
    line1, = ax.plot(data_points[:,0],np.real(curs[:,0]))
    line1.set_label("DC")
    line2, = ax.plot(data_points[:,0],-2.0*np.imag(curs[:,2]))
    line2.set_label("AC sine amplitude, $\omega = 2V$")
    ax.set_xlabel("Gate/$t_{AA}$",fontsize=14)
    ax.set_ylabel("Current/$t_{AA}$, left lead",fontsize=14)
    ax.legend(fontsize=13)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    
    plt.show()

if sys.argv[1] == "nx_ne":

    nn_ne_data_path = "../Data/NN_ne_Sawtooth_tAA135_4_gate100:-4.5:3.0_bias0.5_U-0.3_tLS5.3_VCL1.0_TL0"
    ns_ne_data_path = "../Data/NS_ne_Sawtooth_tAA135_4_gate100:-4.5:3.0_bias0.5_U-0.3_tLS5.3_VCL1.0_DeltaR1.00_TL0"
    if len(sys.argv) > 3:
        nn_ne_data_path = sys.argv[2]
        ns_ne_data_path = sys.argv[3]
    
    
    
    # Normal-normal junction data
    data_points_list_file = open(nn_ne_data_path + '/data_points.csv')
    #
    csvreader = csv.reader(data_points_list_file)
    variables = []
    variables = next(csvreader)
    data_points_list_file.close()
    folders = nn_ne_data_path.split("/")
    folder_name = folders[-1]
    if folders[-1] == "":
        folder_name = folders[-2]
    
    nn_data_points = genfromtxt(nn_ne_data_path +\
            '/data_points.csv',delimiter=';',skip_header=1)[:,1:]
    num_points,num_vars = nn_data_points.shape
    
    params = parse_parameter_file(nn_ne_data_path+'/parameters_const.cfg')
    
    hdf5 = False
    if exists(nn_ne_data_path+"/1.h5"):
        hdf5 = True
    else:
        hdf5 = False

    U = float(params["U"])
    bias = float(params["bias"])
    gate = float(params["gate"])
    tLS = float(params["tLS"])
    tRS = float(params["tRS"])
    TL = float(params["TL"])
    TR = float(params["TR"])
    tL = float(params["tL"])
    tR = float(params["tR"])
    VCL = float(params["VCL"])
    VCR = float(params["VCR"])
    
    n_sites = 0
    f = h5py.File(nn_ne_data_path + "/"+str(1)+".h5","r")
    Gl_mat = np.matrix(f.get("data/Gl0_time_r"))
    f.close()
    n_sites = int(Gl_mat.shape[0]/2)-2
    print(n_sites)
    
    nn_Gl_times = np.zeros((num_points,2*(n_sites+2),2*(n_sites+2)),dtype=complex)
    nn_curs = np.zeros((num_points),dtype=complex)
    nn_pnums = np.zeros((num_points,n_sites),dtype=complex)
    nn_pairs = np.zeros((num_points,n_sites),dtype=complex)
    
    not_found = []
    for i in range(num_points):
        if hdf5:
            f = h5py.File(nn_ne_data_path + "/"+str(i+1)+".h5","r")
            if "data/current_r" in f:
                # Load the lenner Green's function
                Gl_path = "data/Gl"+str(0)+"_time"
                Gl_mat = np.matrix(f.get(Gl_path+"_r")).astype(np.complex128)
                Gl_mat += 1.0j*np.matrix(f.get(Gl_path+"_i")).astype(np.complex128)
                print(Gl_mat.shape)
                nn_Gl_times[i,:,:] = Gl_mat
                nn_curs[i] = -2.0*tLS*nn_Gl_times[i,0,2]+2.0*tLS*nn_Gl_times[i,2,0]
                for j in range(n_sites):
                    nn_pnums[i,j] = -2.0j*nn_Gl_times[i,2*(j+1),2*(j+1)]
                    nn_pairs[i,j] = -1.0j*nn_Gl_times[i,2*(j+1),2*(j+1)+1]
            else:
                not_found.append(i)
            f.close()
    
    print(not_found)

    # Normal-superconducting junction data
    data_points_list_file = open(ns_ne_data_path + '/data_points.csv')
    csvreader = csv.reader(data_points_list_file)
    variables = []
    variables = next(csvreader)
    data_points_list_file.close()
    folders = ns_ne_data_path.split("/")
    folder_name = folders[-1]
    if folders[-1] == "":
        folder_name = folders[-2]
    
    ns_data_points = genfromtxt(ns_ne_data_path +\
            '/data_points.csv',delimiter=';',skip_header=1)[:,1:]
    num_points,num_vars = ns_data_points.shape
    
    params = parse_parameter_file(ns_ne_data_path+'/parameters_const.cfg')
    
    hdf5 = False
    if exists(ns_ne_data_path+"/1.h5"):
        hdf5 = True
    else:
        hdf5 = False

    U = float(params["U"])
    bias = float(params["bias"])
    gate = float(params["gate"])
    tLS = float(params["tLS"])
    tRS = float(params["tRS"])
    TL = float(params["TL"])
    TR = float(params["TR"])
    tL = float(params["tL"])
    tR = float(params["tR"])
    VCL = float(params["VCL"])
    VCR = float(params["VCR"])
    
    n_sites = 0
    f = h5py.File(ns_ne_data_path + "/"+str(1)+".h5","r")
    Gl_mat = np.matrix(f.get("data/Gl0_time_r"))
    f.close()
    n_sites = int(Gl_mat.shape[0]/2)-2
    
    ns_Gl_times = np.zeros((num_points,2*(n_sites+2),2*(n_sites+2)),dtype=complex)
    ns_curs = np.zeros((num_points),dtype=complex)
    ns_pnums = np.zeros((num_points,n_sites),dtype=complex)
    ns_pairs = np.zeros((num_points,n_sites),dtype=complex)
    
    not_found = []
    for i in range(num_points):
        if hdf5:
            f = h5py.File(ns_ne_data_path + "/"+str(i+1)+".h5","r")
            if "data/current_r" in f:
                # Load the lenser Green's function
                Gl_path = "data/Gl"+str(0)+"_time"
                Gl_mat = np.matrix(f.get(Gl_path+"_r")).astype(np.complex128)
                Gl_mat += 1.0j*np.matrix(f.get(Gl_path+"_i")).astype(np.complex128)
                ns_Gl_times[i,:,:] = Gl_mat
                ns_curs[i] = -2.0*tLS*ns_Gl_times[i,0,2]+2.0*tLS*ns_Gl_times[i,2,0]
                for j in range(n_sites):
                    ns_pnums[i,j] = -2.0j*ns_Gl_times[i,2*(j+1),2*(j+1)]
                    ns_pairs[i,j] = -1.0j*ns_Gl_times[i,2*(j+1),2*(j+1)+1]
            else:
                not_found.append(i)
            f.close()
    
    print(not_found)
    
    fig, ax = plt.subplots()
    
    line1, = ax.plot(nn_data_points[:,0],np.real(nn_curs))
    line1.set_label("NN")
    line2, = ax.plot(ns_data_points[:,0],np.real(ns_curs))
    line2.set_label("NS")
    ax.set_xlabel("Gate/$t_{AA}$",fontsize=14)
    ax.set_ylabel("Current/$t_{AA}$, left lead",fontsize=14)
    ax.legend(fontsize=13)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    
    plt.show()



