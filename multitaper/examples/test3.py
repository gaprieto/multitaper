# Testing DPSS codes. 

import numpy as np
import matplotlib.pyplot as plt
import multitaper.utils  as utils
from multitaper import MTSine, MTSpec

#------------------------------------------------
# Define filename and path
#------------------------------------------------

fname = 'PASC.dat'
print('fname ', fname)


#------------------------------------------------
# Define desired parameters
#------------------------------------------------

nw    = 4.0
kspec = 7

#------------------------------------------------
# Load the data
#------------------------------------------------

x    = utils.get_data(fname)
npts = np.shape(x)[0]
dt = 1.0 
t  = np.arange(npts)*dt

#------------------------------------------------
# Get MTSPEC
#     Get spectrum
#     Get QI spectrum
#------------------------------------------------

print('------ Calculating Thomson Multitaper --------')
psd    = MTSpec(x,nw,kspec,dt,iadapt=2)
print('----------------------------------------------')

freq ,spec       = psd.rspec()

fig = plt.figure()
ax = fig.add_subplot(3,1,1)
ax.plot(t/3600,x)
ax = fig.add_subplot(3,2,3)
ax.loglog(freq,psd.sk[0:psd.nf,0])
ax.set_ylim(1e0,1e9)
ax.set_xlim(1e-5,1e0)
ax = fig.add_subplot(3,2,4)
ax.loglog(freq,spec,'r')
ax.set_ylim(1e0,1e9)
ax.set_xlim(1e-5,1e0)

plt.show()

