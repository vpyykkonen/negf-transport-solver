import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re

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

matrix_path = sys.argv[1]
output_name = sys.argv[2]

matrix = read_matrix(matrix_path)
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
plt.show()
#plt.savefig(output_name)

