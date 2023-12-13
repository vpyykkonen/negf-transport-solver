import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from os import path
import csv
from scipy.fftpack import fft, ifft

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

n_gates = 100
n_phases = 30
gates = np.linspace(-4.5,3.0,n_gates)
phases = np.linspace(0.5*np.pi,0.99*np.pi,n_phases)
curs_gate_phase = np.zeros([n_gates,n_phases])

#fig = plt.figure()
#ax11 = fig.add_subplot(131)
#ax12 = fig.add_subplot(132)
#ax13 = fig.add_subplot(133)

crit_curs_gate = np.zeros(n_gates)
curs_phase_pi2 = np.zeros(n_gates)
curs_fft = np.zeros([n_gates,n_phases],dtype=complex)

for i in range(n_gates):
    curs_phases = np.zeros(n_phases)
    for j in range(n_phases):
        curs_phases[j] = read_matrix(
            data_point_path + '/'+\
                str(i+1+j*n_gates)+\
                '/Data/current.csv')[0,0]
    curs_phases_fft = fft(curs_phases)
    #ax11.plot(curs_phases)
    #ax12.plot(np.real(curs_phases_fft))
    #ax13.plot(np.imag(curs_phases_fft))
    crit_curs_gate[i] = np.max(np.abs(curs_phases))
    curs_phase_pi2[i] = curs_phases[int(n_phases/4)-1]
    curs_fft[i,:] = curs_phases_fft


    #plt.plot(phases,curs_phases)
    #curs_gate_phase[i,:] = curs_phases
fig2,ax2 = plt.subplots()
ax2.plot(gates,crit_curs_gate)

#fig3,ax3 = plt.subplots()
#ax3.plot(gates,curs_phase_pi2)

#fig4 = plt.figure()
#ax41 = fig4.add_subplot(121)
#ax41.plot(gates,np.real(curs_fft[:,0]))
#ax42 = fig4.add_subplot(122)
#ax42.plot(gates,np.imag(curs_fft[:,0]))
#
#for i in range(1,10):
#    fig4 = plt.figure()
#    ax41 = fig4.add_subplot(121)
#    ax42 = fig4.add_subplot(122)
#    
#    ax41.plot(gates,2*np.real(curs_fft[:,i]))
#    ax42.plot(gates,-2*np.imag(curs_fft[:,i]))
#    

plt.show()




