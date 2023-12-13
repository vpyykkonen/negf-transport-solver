from matplotlib import pyplot as plt
import numpy as np

def plot_matrix(matrix):
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
