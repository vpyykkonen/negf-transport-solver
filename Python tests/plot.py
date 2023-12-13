import numpy as np
import matplotlib.pyplot as plt

a = -0.00233929-0.0358049j
b = -0.00182376-0.03572112j

ts = np.linspace(-3,3,100)

fs = a*np.exp(1.0j*2*np.pi*ts)+b*np.exp(-1.0j*2*np.pi*ts)

fs_arg = np.angle(fs)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ts,fs_arg)
plt.show()
