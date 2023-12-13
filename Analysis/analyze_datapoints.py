import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from os import path
import csv
import os
import h5py

from matplotlib.widgets import Slider

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
        



if len(sys.argv) == 1:
    sys.exit("Path to data set was not given.")

data_point_path = sys.argv[1]
#data_point_path = './Test/NS-non-equilibrium_Test6'
data_points_list_file = open(data_point_path + '/data_points.csv')
#
csvreader = csv.reader(data_points_list_file)
variables = []
variables = next(csvreader)
data_points_list_file.close()
folders = data_point_path.split("/")
folder_name = folders[-1]
if folders[-1] == "":
    folder_name = folders[-2]
#
#rows = []
#for row in csvreader:
#    rows.append(row)
#print(rows)
#

def parse_parameter_file(path):
    params = {}
    with open(path) as f:
        for line in f:
            re.sub(r'#.*','',line)
            if("=" in line):
                (key,val) = line.split("=",1)
                params[key.strip()] = val.strip()
    return params




params = parse_parameter_file(data_point_path+'/parameters_const.cfg')
geom = parse_parameter_file(data_point_path+'/geometry_const.cfg')
info = {}
hdf5 = False
if exists(data_point_path+"/1.h5"):
    hdf5 = True
else:
    hdf5 = False

if hdf5:
    f = h5py.File(data_point_path+"/1.h5",'r')
    info = dict(f.attrs.items())
    f.close()
else:
    info = parse_parameter_file(data_point_path+'/1/data_point_info.txt')

equilibrium = (int(info["equilibrium"]) == 1)
lead_config = info["lead_config"]
#
U = float(params["U"])
#bias = float(params["bias"])
#gate = float(params["gate"])
#tLS = float(params["tLS"])
#tRS = float(params["tRS"])
#TL = float(params["TL"])
#TR = float(params["TR"])
#tL = float(params["tL"])
#tR = float(params["tR"])
#VCL = float(params["VCL"])
#VCR = float(params["VCR"])
#ieta = parse_complex(params("ieta"))
#
#consts = ["bias","gate","U","tLS","tRS","VCL","VCR","TL","TR","tL","tR","ieta"]
#variables = info["variables"].split(",")
#for var in variables:
#    consts.remove(var)
tAA = -1.0
tAB = np.sqrt(2)*tAA
tLS = 5.3
tRS = 5.3

# Creates the directories to save figures
print_string = folder_name

if not os.path.isdir("Figures/"+print_string):
    os.makedirs("Figures/"+print_string)

print(params)
print(geom)

#dim = int(geom["dim"])
#n_unitcells = list(map(int,geom["n_unitcells"].split(',')))
#unitcell_sites = int(geom["unitcell_sites"])
#
#n_removed_left = 0
#removed_site_strs = geom["edge"].split(';')
##if removed_sites[0]:
#removed_left = removed_site_strs[0].split(',')
#n_removed_left = len(removed_left)
#
#print(n_removed_left)
#
#
#intra_uc_hops = []
#inter_uc_hops = []
#for n in range(int(geom["n_hoppings"])):
#    hop_str = geom["hopping"+str(n+1)]
#    sites_str,hopping_str = hop_str.split('=')
#    hopping = parse_cppcomplex(hopping_str)
#    target_uc,source_orb,target_orb = map(int,sites_str.split(';'))
#
#    source_idx = source_orb
#    target_idx = target_uc*unitcell_sites + target_orb
#    if not equilibrium and lead_config == "SS":
#        source_idx += 1
#        target_idx += 1
#
#    if target_uc != 0:
#        inter_uc_hops.append((source_idx,target_idx,hopping))
#    else:
#        intra_uc_hops.append((source_idx,target_idx,hopping))
#
#print(*intra_uc_hops)
#print(*inter_uc_hops)


data_points = genfromtxt(data_point_path +\
        '/data_points.csv',delimiter=';',skip_header=1)[:,1:]
num_points,num_vars = data_points.shape

n_sites = 0
n_harmonics = 0
rows = n_harmonics
cols = n_harmonics
for i in range(num_points):
    if not hdf5:
        if path.exists(data_point_path + \
                "/"+str(i+1)+"/Data/particle_number.csv"):
            rows,cols = read_matrix(data_point_path +\
                "/"+str(i+1)+"/Data/particle_number.csv").shape
    else:
        f = h5py.File(data_point_path + "/"+str(i+1)+".h5","r")
        if "data/particle_number_r" in f:
            rows,cols = np.array(f.get("data/particle_number_r")).shape
        f.close()
    n_sites = max(n_sites,int(rows))
    n_harmonics = max(n_harmonics,int(cols-1))

print(n_sites)
print(n_harmonics)


curs = np.zeros((num_points,n_harmonics+1),dtype=complex)
curs_Gl = np.zeros((num_points,n_harmonics+1),dtype=complex)
pnums = np.zeros((num_points,n_sites,n_harmonics+1),dtype=complex)
pnums_Gl = np.zeros((num_points,n_sites,n_harmonics+1),dtype=complex)
pair_expects = np.zeros((num_points,n_sites,2*n_harmonics+1),dtype=complex)
pair_expects_Gl = np.zeros((num_points,n_sites,2*n_harmonics+1),dtype=complex)

Gl_times = np.zeros((num_points,n_harmonics+1,2*(n_sites+2),2*(n_sites+2)),dtype=complex)
if lead_config == "closed":
    Gl_times = np.zeros((num_points,n_harmonics+1,2*n_sites,2*n_sites),dtype=complex)

if n_harmonics == 0:
    Gl_times = np.zeros((num_points,1,2*(n_sites+2),2*(n_sites+2)),dtype=complex)
    if lead_config == "closed":
        Gl_times = np.zeros((num_points,1,2*n_sites,2*n_sites),dtype=complex)

not_found = []
for i in range(num_points):
    if hdf5:
        f = h5py.File(data_point_path + "/"+str(i+1)+".h5","r")
        if "data/current_r" in f:
            curs_mat = np.matrix(f.get("data/current_r")).astype(np.complex128)
            curs_mat += 1.0j*np.matrix(f.get("data/current_i"))

            pnums_mat = np.matrix(f.get("data/particle_number_r")).astype(np.complex128)
            pnums_mat += 1.0j*np.matrix(f.get("data/particle_number_i"))

            pairs_mat = np.matrix(f.get("data/pair_expectation_r")).astype(np.complex128)
            pairs_mat += 1.0j*np.matrix(f.get("data/pair_expectation_i"))

            curs_rows,curs_cols = curs_mat.shape
            pnums_rows,pnums_cols = pnums_mat.shape
            pairs_rows,pairs_cols = pairs_mat.shape

            curs[i,:curs_rows] = curs_mat.reshape((1,curs_rows))[:,:curs_rows]
            pnums[i,:,:pnums_cols] = pnums_mat
            pair_expects[i,:,:pairs_cols] = pairs_mat
            for n in range(n_harmonics+1):
                Gl_path = "data/Gl"+str(n)+"_time"
                Gl_mat = np.matrix(f.get(Gl_path+"_r")).astype(np.complex128)
                Gl_mat += 1.0j*np.matrix(f.get(Gl_path+"_i")).astype(np.complex128)
                Gl_times[i,n,:,:] = Gl_mat

            for n in range(n_harmonics+1):
                #if(n == 0):
                #    #curs_Gl[i,0] = -4.0*tLS*(-np.conj(Gl_times[i,1,2,0]))
                #    curs_Gl[i,0] = -2.0*tLS*Gl_times[i,1,0,2]-2.0*tLS*np.conj(Gl_times[i,1,0,2])
                #elif(n < n_harmonics):
                #    curs_Gl[i,n] = -2.0*tLS*Gl_times[i,n+1,0,2] + 2.0*tLS*Gl_times[i,n-1,2,0]
                #if(n == 0):
                #    #curs_Gl[i,0] = -4.0*tLS*(-np.conj(Gl_times[i,1,2,0]))
                #    curs_Gl[i,0] = -2.0*tRS*Gl_times[i,1,2*n_sites,2*(n_sites+1)]-2.0*tLS*np.conj(Gl_times[i,1,2*n_sites,2*(n_sites+1)])
                #elif(n < n_harmonics):
                #    curs_Gl[i,n] = -2.0*tRS*Gl_times[i,n+1,2*n_sites,2*(n_sites+1)] + 2.0*tLS*Gl_times[i,n-1,2*(n_sites+1),2*n_sites]
                #curs_Gl[i,n] = -2.0*tRS*Gl_times[i,n,2*n_sites,2*(n_sites+1)] + 2.0*tRS*Gl_times[i,n,2*(n_sites+1),2*n_sites]
                curs_Gl[i,n] = -2.0*tAA*Gl_times[i,n,2,6] + 2.0*tAA*Gl_times[i,n,6,2]
                #curs_Gl[i,n] += -2.0*tAB*Gl_times[i,n,4,6] + 2.0*tAB*Gl_times[i,n,6,4]
                for j in range(n_sites):
                    if lead_config != "closed":
                        pnums_Gl[i,j,n] = -2.0j*Gl_times[i,n,2*(j+1),2*(j+1)]
                    else:
                        pnums_Gl[i,j,n] = -2.0j*Gl_times[i,n,2*j,2*j]
                    if(n == 0):
                        if lead_config != "closed":
                            pair_expects_Gl[i,j,0] = -1.0j*Gl_times[i,0,2*(j+1),2*(j+1)+1]
                        else:
                            pair_expects_Gl[i,j,0] = -1.0j*Gl_times[i,0,2*j,2*j+1]
                    else:
                        # Positive harmonic
                        pair_expects_Gl[i,j,2*n-1] = -1.0j*Gl_times[i,n,2*(j+1),2*(j+1)+1]
                        # Negative harmonic
                        pair_expects_Gl[i,j,2*n] = 1.0j*np.conj(Gl_times[i,n,2*(j+1)+1,2*(j+1)])
        else:
            not_found.append(i)
        f.close()
    else:
        if path.exists(data_point_path + \
                "/"+str(i+1)+"/Data/current.csv"):
            curs_mat = read_matrix(
                data_point_path + \
                    "/"+str(i+1)+"/Data/current.csv")
            pnums_mat = read_matrix(
                data_point_path + \
                    "/"+str(i+1)+"/Data/particle_number.csv")
            pairs_mat = read_matrix(
                data_point_path + \
                    "/"+str(i+1)+"/Data/pair_expectation.csv")
            # Get the number of harmonics in the matrix
            curs_rows,curs_cols = curs_mat.shape
            pnums_rows,pnums_cols = pnums_mat.shape
            pairs_rows,pairs_cols = pairs_mat.shape

            curs[i,:curs_rows] = curs_mat.reshape((1,curs_rows))[:,:curs_rows]
            pnums[i,:,:pnums_cols] = pnums_mat
            pair_expects[i,:,:pairs_cols] = pairs_mat
        else:
            not_found.append(i)

print(not_found)

curs = np.delete(curs,not_found,0)
pnums = np.delete(pnums,not_found,0)
pair_expects = np.delete(pair_expects,not_found,0)
curs_Gl = np.delete(curs_Gl,not_found,0)
pnums_Gl = np.delete(pnums_Gl,not_found,0)
pair_expects_Gl = np.delete(pair_expects_Gl,not_found,0)
data_points = np.delete(data_points,not_found,0)

def plot_curs_vs_gate(data_points,curs):
    n_harmonics = curs.shape[1] - 1
    fig1,ax1 = plt.subplots()
    ax1.set_xlim([data_points[0,0]-0.2,data_points[-1,0]+0.2])
    ylim_max = np.amax(2.0*np.abs(curs))+0.01
    ylim_min = np.amin(-2.0*np.abs(curs))-0.01
    ax1.set_ylim([ylim_min,ylim_max])
    
    line11, = ax1.plot(data_points[:,0],np.real(curs[:,0]))
    line12, = ax1.plot(data_points[:,0],np.imag(curs[:,0]))
    ax1.set_xlabel("gate")
    ax1.set_ylabel("Current")
    line11.set_label("cos(2Vnt)")
    line12.set_label("sin(2Vnt)")
    ax1.legend()
    
    sharmonic = 0
    def sharmonic_update(val):
        if val != 0:
            line11.set_ydata(2.0*np.real(curs[:,int(val)]))
            line12.set_ydata(-2.0*np.imag(curs[:,int(val)]))
        else:
            line11.set_ydata(np.real(curs[:,int(val)]))
            line12.set_ydata(np.imag(curs[:,int(val)]))
        #ylim_max = max(np.amax(-2.0*np.imag(curs[:,val])),np.amax(2.0*np.real(curs[:,val])))
        #ylim_min = min(np.amin(-2.0*np.imag(curs[:,val])),np.amin(2.0*np.real(curs[:,val])))
        #ax1.set_ylim([ylim_min-0.01,ylim_max+0.01])
        fig1.canvas.draw_idle()
    
    if n_harmonics > 0:
        ax1_pos = ax1.get_position()
        ax_sharmonic = fig1.add_axes([ax1_pos.x0,ax1_pos.y0 + ax1_pos.height + 0.005,ax1_pos.width,0.03])
        sharmonic = Slider(
            ax = ax_sharmonic,
            label = "harmonic n = ",
            valmin = 0,
            valmax = n_harmonics,
            valstep = 2,
            valinit = 0
        )
        sharmonic.on_changed(sharmonic_update)
    if n_harmonics > 0:
        return fig1,ax1,sharmonic
    return fig1,ax1

def plot_pnums_vs_gate(data_points,pnums):
    n_harmonics = pnums.shape[2] - 1
    n_sites = pnums.shape[1]

    fig1,ax1 = plt.subplots()
    ax1.set_xlim([data_points[0,0]-0.2,data_points[-1,0]+0.2])
    ylim_max = np.amax(2.0*np.abs(pnums))+0.01
    ylim_min = np.amin(-2.0*np.abs(pnums))-0.01
    ax1.set_ylim([ylim_min,ylim_max])
    ax1.set_xlabel("Gate")
    ax1.set_ylabel("Particle number")
    
    line21, = ax1.plot(data_points[:,0],np.real(pnums[:,0,0]))
    line22, = ax1.plot(data_points[:,0],np.imag(pnums[:,0,0]))
    line21.set_label("cos(2Vnt)")
    line22.set_label("sin(2Vnt)")
    ax1.legend()
    
    ax1_pos = ax1.get_position()
    ax_ssite = fig1.add_axes([ax1_pos.x0,ax1_pos.y0+ax1_pos.height+0.005,ax1_pos.width,0.03])
    ssite = Slider(
        ax = ax_ssite,
        label = "site",
        valmin = 1,
        valmax = n_sites,
        valstep = 1,
        valinit = 0
    )
    
    sharmonic = 0
    if(n_harmonics > 0):
        ax1_pos = ax1.get_position()
        ax_sharmonic = fig1.add_axes([ax1_pos.x0,ax1_pos.y0+ax1_pos.height+0.04,ax1_pos.width,0.03])
        sharmonic = Slider(
            ax = ax_sharmonic,
            label = "harmonic n = ",
            valmin = 0,
            valmax = n_harmonics,
            valstep = 2,
            valinit = 0
        )
    
    def pnum_slider_update(val):
        if n_harmonics > 0:
            if sharmonic.val != 0:
                line21.set_ydata(2.0*np.real(pnums[:,int(ssite.val)-1,int(sharmonic.val)]))
                line22.set_ydata(-2.0*np.imag(pnums[:,int(ssite.val)-1,int(sharmonic.val)]))
            else:
                line21.set_ydata(np.real(pnums[:,int(ssite.val)-1,int(sharmonic.val)]))
                line22.set_ydata(np.imag(pnums[:,int(ssite.val)-1 ,int(sharmonic.val)]))
        else:
            line21.set_ydata(np.real(pnums[:,int(ssite.val)-1,0]))
            line22.set_ydata(np.imag(pnums[:,int(ssite.val)-1 ,0]))

    
        fig1.canvas.draw_idle()
    
    
    ssite.on_changed(pnum_slider_update)
    if n_harmonics > 0:
        sharmonic.on_changed(pnum_slider_update)
        return fig1,ax1,ssite,sharmonic
    return fig1,ax1,ssite


def plot_pnums_vs_site(data_points,pnums):
    n_harmonics = pnums.shape[2] - 1
    n_sites = pnums.shape[1]

    fig1,ax1 = plt.subplots()
    sites = np.linspace(1,n_sites,n_sites)
    ax1.set_xlim([sites[0]-0.5,sites[-1]+0.5])
    ylim_max = np.amax(2.0*np.abs(pnums))+0.01
    ylim_min = np.amin(-2.0*np.abs(pnums))-0.01
    ax1.set_ylim([ylim_min,ylim_max])
    ax1.set_xlabel("Site")
    ax1.set_ylabel("Particle number")
    
    line21, = ax1.plot(sites,np.real(pnums[0,:,0]))
    line22, = ax1.plot(sites,np.imag(pnums[0,:,0]))
    line21.set_label("cos(2Vnt)")
    line22.set_label("sin(2Vnt)")
    ax1.legend()
    
    ax1_pos = ax1.get_position()
    ax_sgate = fig1.add_axes([ax1_pos.x0,ax1_pos.y0+ax1_pos.height+0.005,ax1_pos.width,0.03])
    sgate = Slider(
        ax = ax_sgate,
        label = "gate",
        valmin = 0,
        valmax = data_points.shape[0]-1,
        valstep = 1,
        valinit = 0
    )
    
    sharmonic = 0
    if(n_harmonics > 0):
        ax1_pos = ax1.get_position()
        ax_sharmonic = fig1.add_axes([ax1_pos.x0,ax1_pos.y0+ax1_pos.height+0.04,ax1_pos.width,0.03])
        sharmonic = Slider(
            ax = ax_sharmonic,
            label = "harmonic n = ",
            valmin = 0,
            valmax = n_harmonics,
            valstep = 2,
            valinit = 0
        )
    
    def pnum_slider_update(val):
        if n_harmonics > 0:
            if sharmonic.val != 0:
                line21.set_ydata(2.0*np.real(pnums[int(sgate.val),:,int(sharmonic.val)]))
                line22.set_ydata(-2.0*np.imag(pnums[int(sgate.val),:,int(sharmonic.val)]))
            else:
                line21.set_ydata(np.real(pnums[int(sgate.val),:,0]))
                line22.set_ydata(np.imag(pnums[int(sgate.val),:,0]))
        else:
            line21.set_ydata(np.real(pnums[int(sgate.val),:,0]))
            line22.set_ydata(np.imag(pnums[int(sgate.val),:,0]))
    
        fig1.canvas.draw_idle()
    
    
    sgate.on_changed(pnum_slider_update)
    if n_harmonics > 0:
        sharmonic.on_changed(pnum_slider_update)
        return fig1,ax1,sgate,sharmonic
    return fig1,ax1,sgate

def plot_pairs_vs_gate(data_points,pairs):
    n_harmonics = pairs.shape[2] - 1
    n_sites = pairs.shape[1]

    fig1,ax1 = plt.subplots()
    ax1.set_xlim([data_points[0,0]-0.2,data_points[-1,0]+0.2])
    ylim_max = np.amax(2.0*np.abs(pairs))+0.01
    ylim_min = np.amin(-2.0*np.abs(pairs))-0.01
    ax1.set_ylim([ylim_min,ylim_max])
    ax1.set_xlabel("Gate")
    ax1.set_ylabel("Pair expectation")
    
    line1, = ax1.plot(data_points[:,0],np.real(pairs[:,0,0]))
    line2, = ax1.plot(data_points[:,0],np.imag(pairs[:,0,0]))
    line3, = ax1.plot(data_points[:,0],np.real(pairs[:,0,0]))
    line4, = ax1.plot(data_points[:,0],np.imag(pairs[:,0,0]))
    line1.set_label("Positive, real")
    line2.set_label("Positive, imag")
    line3.set_label("Negative, real")
    line4.set_label("Negative, imag")
    ax1.legend()
    
    ax1_pos = ax1.get_position()
    ax_ssite = fig1.add_axes([ax1_pos.x0,ax1_pos.y0+ax1_pos.height+0.005,ax1_pos.width,0.03])
    ssite = Slider(
        ax = ax_ssite,
        label = "site",
        valmin = 1,
        valmax = n_sites,
        valstep = 1,
        valinit = 0
    )
    
    sharmonic = 0
    if(n_harmonics > 0):
        ax1_pos = ax1.get_position()
        ax_sharmonic = fig1.add_axes([ax1_pos.x0,ax1_pos.y0+ax1_pos.height+0.04,ax1_pos.width,0.03])
        sharmonic = Slider(
            ax = ax_sharmonic,
            label = "harmonic n = ",
            valmin = 0,
            valmax = n_harmonics,
            valstep = 2,
            valinit = 0
        )
    
    def pair_slider_update(val):
        if n_harmonics > 0:
            if int(sharmonic.val) != 0:
                line1.set_ydata(np.real(pairs[:,int(ssite.val)-1,2*int(sharmonic.val)-1]))
                line3.set_ydata(np.imag(pairs[:,int(ssite.val)-1,2*int(sharmonic.val)-1]))
                line3.set_ydata(np.real(pairs[:,int(ssite.val)-1,2*int(sharmonic.val)]))
                line4.set_ydata(np.imag(pairs[:,int(ssite.val)-1,2*int(sharmonic.val)]))
            else:
                line1.set_ydata(np.real(pairs[:,int(ssite.val)-1,0]))
                line2.set_ydata(np.imag(pairs[:,int(ssite.val)-1,0]))
                line3.set_ydata(np.real(pairs[:,int(ssite.val)-1,0]))
                line4.set_ydata(np.imag(pairs[:,int(ssite.val)-1,0]))
        else:
            #line1.set_ydata(np.real(pairs[:,int(ssite.val)-1,0]))
            #line2.set_ydata(np.imag(pairs[:,int(ssite.val)-1,0]))
            #line3.set_ydata(np.real(pairs[:,int(ssite.val)-1,0]))
            #line4.set_ydata(np.imag(pairs[:,int(ssite.val)-1,0]))
            line1.set_ydata(np.abs(pairs[:,int(ssite.val)-1,0]))
            line2.set_ydata(np.angle(pairs[:,int(ssite.val)-1,0])/100)
            line3.set_ydata(np.real(pairs[:,int(ssite.val)-1,0]))
            line4.set_ydata(np.imag(pairs[:,int(ssite.val)-1,0]))
    
    
    ssite.on_changed(pair_slider_update)
    if(n_harmonics > 0):
        sharmonic.on_changed(pair_slider_update)
        return fig1,ax1,ssite,sharmonic
    return fig1,ax1,ssite

def plot_pairs_vs_site(data_points,pairs):
    n_harmonics = pairs.shape[2] - 1
    n_sites = pairs.shape[1]

    fig1,ax1 = plt.subplots()
    sites = np.linspace(1,n_sites,n_sites)
    ax1.set_xlim([sites[0]-0.5,sites[-1]+0.5])
    ylim_max = np.amax(2.0*np.abs(pairs))+0.01
    ylim_min = np.amin(-2.0*np.abs(pairs))-0.01
    ax1.set_ylim([ylim_min,ylim_max])
    ax1.set_xlabel("Site")
    ax1.set_ylabel("Pair expectation")
    
    line1, = ax1.plot(sites,np.real(pairs[0,:,0]))
    line2, = ax1.plot(sites,np.imag(pairs[0,:,0]))
    line3, = ax1.plot(sites,np.real(pairs[0,:,0]))
    line4, = ax1.plot(sites,np.imag(pairs[0,:,0]))
    line1.set_label("Positive, real")
    line2.set_label("Positive, imag")
    line3.set_label("Negative, real")
    line4.set_label("Negative, imag")
    ax1.legend()
    
    ax1_pos = ax1.get_position()
    ax_sgate = fig1.add_axes([ax1_pos.x0,ax1_pos.y0+ax1_pos.height+0.005,ax1_pos.width,0.03])
    sgate = Slider(
        ax = ax_sgate,
        label = "gate",
        valmin = 0,
        valmax = data_points.shape[0]-1,
        valstep = 1,
        valinit = 0
    )
    
    sharmonic = 0
    if(n_harmonics > 0):
        ax1_pos = ax1.get_position()
        ax_sharmonic = fig1.add_axes([ax1_pos.x0,ax1_pos.y0+ax1_pos.height+0.04,ax1_pos.width,0.03])
        sharmonic = Slider(
            ax = ax_sharmonic,
            label = "harmonic n = ",
            valmin = 0,
            valmax = n_harmonics,
            valstep = 2,
            valinit = 0
        )
    
    def pair_slider_update(val):
        if n_harmonics > 0:
            if sharmonic.val != 0:
                line1.set_ydata(np.real(pairs[int(sgate.val),:,2*int(sharmonic.val)-1]))
                line2.set_ydata(np.imag(pairs[int(sgate.val),:,2*int(sharmonic.val)-1]))
                line3.set_ydata(np.real(pairs[int(sgate.val),:,2*int(sharmonic.val)]))
                line4.set_ydata(np.imag(pairs[int(sgate.val),:,2*int(sharmonic.val)]))
            else:
                line1.set_ydata(np.real(pairs[int(sgate.val),:,0]))
                line2.set_ydata(np.imag(pairs[int(sgate.val),:,0]))
                line3.set_ydata(np.real(pairs[int(sgate.val),:,0]))
                line4.set_ydata(np.imag(pairs[int(sgate.val),:,0]))
        else:
            line1.set_ydata(np.abs(pairs[int(sgate.val),:,0]))
            line2.set_ydata(np.angle(pairs[int(sgate.val),:,0])/100)
            line3.set_ydata(np.real(pairs[int(sgate.val),:,0]))
            line4.set_ydata(np.imag(pairs[int(sgate.val),:,0]))
    
        fig1.canvas.draw_idle()
    
    
    sgate.on_changed(pair_slider_update)
    if n_harmonics > 0:
        sharmonic.on_changed(pair_slider_update)
        return fig1,ax1,sgate,sharmonic
    return fig1,ax1,sgate

n_harmonics = curs.shape[1] - 1

curvg_fig = 0
curvg_ax = 0
curvg_sharmonic = 0
if n_harmonics > 0 :
    curvg_fig,curvg_ax,curvg_sharmonic = plot_curs_vs_gate(data_points,curs_Gl)
else:
    curvg_fig,curvg_ax = plot_curs_vs_gate(data_points,curs_Gl)

pnumvg_fig = 0
pnumvg_ax = 0
pnumvg_ssite = 0
pnumvg_sharmonics = 0

if n_harmonics > 0:
    pnumvg_fig,pnumvg_ax,pnumvg_ssite,pnumvg_sharmonics = plot_pnums_vs_gate(data_points,pnums_Gl)
else:
    pnumvg_fig,pnumvg_ax,pnumvg_ssite = plot_pnums_vs_gate(data_points,pnums_Gl)

pnumvs_fig = 0
pnumvs_ax = 0
pnumvs_sgate = 0
pnumvs_sharmonics = 0

if n_harmonics > 0:
    pnumvs_fig,pnumvs_ax,pnumvs_sgate,pnumvs_sharmonics = plot_pnums_vs_site(data_points,pnums_Gl)
else:
    pnumvs_fig,pnumvs_ax,pnumvs_ssite = plot_pnums_vs_site(data_points,pnums_Gl)

pairvg_fig = 0
pairvg_ax = 0
pairvg_ssite = 0
pairvg_sharmonics = 0

if n_harmonics > 0:
    pairvg_fig,pairvg_ax,pairvg_ssite,pairvg_sharmonics = plot_pairs_vs_gate(data_points,pair_expects_Gl)
else:
    pairvg_fig,pairvg_ax,pairvg_ssite = plot_pairs_vs_gate(data_points,pair_expects_Gl)

pairvs_fig = 0
pairvs_ax = 0
pairvs_sgate = 0
pairvs_sharmonics = 0

if n_harmonics > 0:
    pairvs_fig,pairvs_ax,pairvs_sgate,pairvs_sharmonics = plot_pairs_vs_site(data_points,pair_expects_Gl)
else:
    pairvs_fig,pairvs_ax,pairvs_ssite = plot_pairs_vs_site(data_points,pair_expects_Gl)


plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(data_points[:,0],np.real(curs[:,0]))
ax1.set_xlabel(variables[0])
ax1.set_ylabel("DC Current, 1st uc to 2nd") 
plt.savefig("Figures/"+print_string+"/Current0_vs_"+variables[0]+".pdf",format='pdf')


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(data_points[:,0],np.sum(np.abs(pnums[:,:,0]),1))
ax2.set_xlabel(variables[0])
ax2.set_ylabel("Particle number") 
plt.savefig("Figures/"+print_string+"/pnum0_vs_"+variables[0]+".pdf",format='pdf')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
Ucoeff = U
if Ucoeff < 1e-4:
    Ucoeff = 1
ax3.plot(data_points[:,0],Ucoeff*np.amax(np.abs(pair_expects[:,:,0]),1))
ax3.set_xlabel(variables[0])
if U < 1e-4:
    ax3.set_ylabel("Pair expectation value") 
else:
    ax3.set_ylabel("Order parameter") 
plt.savefig("Figures/"+print_string+"/delta0_vs_"+variables[0]+".pdf",format='pdf')

ord_nums = ["first","second","third","fourth","fifth","sixth"]
figs = []
curs_rows, curs_cols = curs.shape
if(n_harmonics > 0):
    #for n in range(1,int(curs_cols/2)):
    for n in range(1,3):
        if(n > 6):
            break
    
        # Current
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax1.plot(data_points[:,0],2.0*np.real(curs[:,2*n]))
        ax1.set_xlabel(variables[0])
        ax1.set_ylabel("Current left lead, %s harmonic $\omega=%d \cdot 2V$, cosine" % (ord_nums[n-1],n)) 
        
        ax1 = fig1.add_subplot(122)
        ax1.plot(data_points[:,0],-2.0*np.imag(curs[:,2*n]))
        ax1.set_xlabel(variables[0])
        ax1.set_ylabel("Current left lead, %s harmonic $\omega=%d \cdot 2V$, sine" % (ord_nums[n-1],n)) 
        plt.tight_layout()
        plt.savefig("Figures/"+print_string+"/Current"+str(n)+"_vs_"+variables[0]+".pdf",format='pdf')

        # Particle number
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax1.plot(data_points[:,0],2.0*np.sum(np.real(pnums[:,:,2*n]),1))
        ax1.set_xlabel(variables[0])
        ax1.set_ylabel("Particle number, %s harmonic $\omega=%d \cdot 2V$, cosine" % (ord_nums[n-1],n)) 
        
        ax1 = fig1.add_subplot(122)
        ax1.plot(data_points[:,0],-2.0*np.sum(np.imag(pnums[:,:,2*n]),1))
        ax1.set_xlabel(variables[0])
        ax1.set_ylabel("Particle number, %s harmonic $\omega=%d \cdot 2V$, sine" % (ord_nums[n-1],n)) 
        plt.tight_layout()
        plt.savefig("Figures/"+print_string+"/pnum"+str(n)+"_vs_"+variables[0]+".pdf",format='pdf')

    
        # Pair expectation
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        Ucoeff = abs(U)
        if Ucoeff < 1e-4:
            Ucoeff = 1
        ax1.plot(data_points[:,0],Ucoeff*np.amax(np.abs(pair_expects[:,:,4*n-1]),1))
        ax1.set_xlabel(variables[0])
        if abs(U) < 1e-4:
            ax1.set_ylabel("Pair expectation %s harmonic $\omega=%d \cdot V$, positive" % (ord_nums[n-1],n) )
        else:
            ax1.set_ylabel("Order parameter %s harmonic $\omega=%d \cdot V$, positive" % (ord_nums[n-1],n) )

        ax1 = fig1.add_subplot(122)
        Ucoeff = U
        if Ucoeff < 1e-4:
            Ucoeff = 1
        ax1.plot(data_points[:,0],Ucoeff*np.amax(np.abs(pair_expects[:,:,4*n]),1))
        ax1.set_xlabel(variables[0])
        if abs(U) < 1e-4:
            ax1.set_ylabel("Pair expectation %s harmonic $\omega=%d \cdot V$, negative" % (ord_nums[n-1],n) )
        else:
            ax1.set_ylabel("Order parameter %s harmonic $\omega=%d \cdot V$, negative" % (ord_nums[n-1],n) )

        plt.tight_layout()
        plt.savefig("Figures/"+print_string+"/delta"+str(n)+"_vs_"+variables[0]+".pdf",format='pdf')

        # A1
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        Ucoeff = abs(U)
        if Ucoeff < 1e-4:
            Ucoeff = 1
        ax1.plot(data_points[:,0],Ucoeff*np.abs(pair_expects[:,0,4*n-1]))
        ax1.set_xlabel(variables[0])
        if abs(U) < 1e-4:
            ax1.set_ylabel("Pair expectation at A1 %s harmonic $\omega=%d \cdot V$, positive" % (ord_nums[n-1],n) )
        else:
            ax1.set_ylabel("Order parameter at A1 %s harmonic $\omega=%d \cdot V$, positive" % (ord_nums[n-1],n) )

        ax1 = fig1.add_subplot(122)
        Ucoeff = U
        if Ucoeff < 1e-4:
            Ucoeff = 1
        ax1.plot(data_points[:,0],Ucoeff*np.abs(pair_expects[:,0,4*n]))
        ax1.set_xlabel(variables[0])
        if abs(U) < 1e-4:
            ax1.set_ylabel("Pair expectation at A1 %s harmonic $\omega=%d \cdot V$, negative" % (ord_nums[n-1],n) )
        else:
            ax1.set_ylabel("Order parameter at A1 %s harmonic $\omega=%d \cdot V$, negative" % (ord_nums[n-1],n) )

        plt.tight_layout()

        # A2
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        Ucoeff = abs(U)
        if Ucoeff < 1e-4:
            Ucoeff = 1
        ax1.plot(data_points[:,0],Ucoeff*np.abs(pair_expects[:,2,4*n-1]))
        ax1.set_xlabel(variables[0])
        if abs(U) < 1e-4:
            ax1.set_ylabel("Pair expectation at A2 %s harmonic $\omega=%d \cdot V$, positive" % (ord_nums[n-1],n) )
        else:
            ax1.set_ylabel("Order parameter at A2 %s harmonic $\omega=%d \cdot V$, positive" % (ord_nums[n-1],n) )

        ax1 = fig1.add_subplot(122)
        Ucoeff = U
        if Ucoeff < 1e-4:
            Ucoeff = 1
        ax1.plot(data_points[:,0],Ucoeff*np.abs(pair_expects[:,2,4*n]))
        ax1.set_xlabel(variables[0])
        if abs(U) < 1e-4:
            ax1.set_ylabel("Pair expectation at A2 %s harmonic $\omega=%d \cdot V$, negative" % (ord_nums[n-1],n) )
        else:
            ax1.set_ylabel("Order parameter at A2 %s harmonic $\omega=%d \cdot V$, negative" % (ord_nums[n-1],n) )

        plt.tight_layout()


plt.show()














