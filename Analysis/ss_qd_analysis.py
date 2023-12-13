import sys
import os
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from os import path
import csv

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
        

if len(sys.argv) == 1:
    sys.exit("Path to data set was not given.")
data_point_path = sys.argv[1]
harmonic = int(sys.argv[2])
row_idx  = int(sys.argv[3])
col_idx  = int(sys.argv[4])

if len(sys.argv)==5:
    pres =1
else:
    pres = int(sys.argv[5])


if(harmonic != -1):
    path_elements = data_point_path.split("/")
    number = path_elements[-1]
    path_name = path_elements[-2]
    if(path_elements[-1] == ""):
        number = path_elements[-2]
        path_name = path_elements[-3]
    
    path = data_point_path + "/Data/Gln"+str(harmonic)+"_time.csv"
    Gln_time = read_matrix(path)
    
    pos_freq = -1.0j*Gln_time[row_idx,col_idx]
    neg_freq = 1.0j*np.conj(Gln_time[col_idx,row_idx])
    
    
    n_ts = 100
    ts = np.linspace(0.0,1.0,n_ts)
    
    
    freq = 2*np.pi*harmonic
    if harmonic == 0:
        freq = 0
    
    signal = pos_freq*np.exp(1.0j*freq*ts)+neg_freq*np.exp(-1.0j*freq*ts)
    if harmonic == 0:
        signal /= 2
    
    if not os.path.isdir("Figures/"+path_name):
        os.makedirs("Figures/"+path_name)
    
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    if pres == 1:
        ax1.plot(ts,np.abs(signal))
        ax2.plot(ts,np.angle(signal))
        ax1.set_xlabel('Time')
        #ax1.set_ylabel(r'$\langle \hat{c}_{QD\downarrow}\hat{c}_{QD\uparrow}\rangle$ Amplitude')
        ax1.set_ylabel(r'$-iG^<_{%d,%d,%d}(t)$ Amplitude' %(harmonic,row_idx,col_idx))
        ax2.set_xlabel('Time')
        ax2.set_ylabel(r'$-iG^<_{%d,%d,%d}(t)$ Phase angle' %(harmonic,row_idx,col_idx))
    if pres == 2:
        ax1.plot(ts,np.real(signal))
        ax2.plot(ts,np.imag(signal))
        ax1.set_xlabel('Time')
        #ax1.set_ylabel(r'$\langle \hat{c}_{QD\downarrow}\hat{c}_{QD\uparrow}\rangle$ Amplitude')
        ax1.set_ylabel(r'$-iG^<_{%d,%d,%d}(t)$ Real part' %(harmonic,row_idx,col_idx))
        ax2.set_xlabel('Time')
        ax2.set_ylabel(r'$-iG^<_{%d,%d,%d}(t)$ Imaginary part' %(harmonic,row_idx,col_idx))
    #ax2.set_ylabel(r'$\langle \hat{c}_{QD\downarrow}\hat{c}_{QD\uparrow}\rangle$ Phase angle')
    plt.tight_layout()
    plt.savefig("Figures/"+path_name+"/Gln"+str(harmonic)+"_"+str(row_idx)+"_"+str(col_idx)+"_vs_time_"+number+".pdf",format='pdf')
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(ts,np.real(signal))
    ax2.plot(ts,np.imag(signal))
    
    #plt.show()
else:
    path_elements = data_point_path.split("/")
    number = path_elements[-1]
    path_name = path_elements[-2]
    if(path_elements[-1] == ""):
        number = path_elements[-2]
        path_name = path_elements[-3]
    harmonic_finite = True
    harmonic = 0
    n_ts = 100
    ts = np.linspace(0.0,1.0,n_ts)
    signal = np.zeros(n_ts,dtype=complex)
    while harmonic_finite:
        path = data_point_path + "/Data/Gln"+str(harmonic)+"_time.csv"
        print("harmonic = ",harmonic)
        if os.path.exists(path):
            harmonic_finite = True
        else:
            harmonic_finite = False
            break
        
        Gln_time = read_matrix(path)
        
        if math.isinf(np.real(Gln_time[row_idx,col_idx])):
            harmonic += 1
            break
        pos_freq = -1.0j*Gln_time[row_idx,col_idx]
        neg_freq = 1.0j*np.conj(Gln_time[col_idx,row_idx])
        print(pos_freq)
        print(neg_freq)

        
        
        
        
        freq = 2*np.pi*harmonic
        if harmonic == 0:
            freq = 0
        
        signal += pos_freq*np.exp(1.0j*freq*ts)+neg_freq*np.exp(-1.0j*freq*ts)

        if harmonic == 0:
            signal /= 2
        harmonic += 1
        
    if not os.path.isdir("Figures/"+path_name):
        os.makedirs("Figures/"+path_name)
    
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    if pres == 1:
        ax1.plot(ts,np.abs(signal))
        ax2.plot(ts,np.angle(signal))
        ax1.set_xlabel('Time')
        #ax1.set_ylabel(r'$\langle \hat{c}_{QD\downarrow}\hat{c}_{QD\uparrow}\rangle$ Amplitude')
        ax1.set_ylabel(r'$-iG^<_{%d,%d}(t)$ Amplitude' %(row_idx,col_idx))
        ax2.set_xlabel('Time')
        ax2.set_ylabel(r'$-iG^<_{%d,%d}(t)$ Phase angle' %(row_idx,col_idx))
    if pres == 2:
        ax1.plot(ts,np.real(signal))
        ax2.plot(ts,np.imag(signal))
        ax1.set_xlabel('Time')
        #ax1.set_ylabel(r'$\langle \hat{c}_{QD\downarrow}\hat{c}_{QD\uparrow}\rangle$ Amplitude')
        ax1.set_ylabel(r'$-iG^<_{%d,%d}(t)$ real part' %(row_idx,col_idx))
        ax2.set_xlabel('Time')
        ax2.set_ylabel(r'$-iG^<_{%d,%d}(t)$ imaginary part' %(row_idx,col_idx))
    #ax2.set_ylabel(r'$\langle \hat{c}_{QD\downarrow}\hat{c}_{QD\uparrow}\rangle$ Phase angle')
    plt.tight_layout()
    plt.savefig("Figures/"+path_name+"/Gl_"+str(row_idx)+"_"+str(col_idx)+"_vs_time_"+number+".pdf",format='pdf')
    
    #fig = plt.figure()
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)
    #ax1.plot(ts,np.real(signal))
    #ax2.plot(ts,np.imag(signal))
    
    #plt.show()



