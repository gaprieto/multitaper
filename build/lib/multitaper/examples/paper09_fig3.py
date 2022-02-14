# Testing DPSS codes. 

import multitaper.mtspec as spec
import numpy as np
import matplotlib.pyplot as plt
import multitaper.utils  as utils

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
# Get MTSINE
#     Get sine psd
#------------------------------------------------

print('----- Calculating Sine Multitaper ------')
sine_psd    = spec.MTSine(x,ntap=0,ntimes=2,fact=1.0,dt=dt)
print('----------------------------------------')
#------------------------------------------------
# Get MTSPEC
#     Get spectrum
#     Get QI spectrum
#------------------------------------------------

print('------ Calculating Thomson Multitaper --------')
psd    = spec.MTSpec(x,nw,kspec,dt,iadapt=0)
print('----------------------------------------------')
print('------ Calculating Quadratic Multitaper ------')
qispec = psd.qiinv()[0]
print('----------------------------------------------')

# Plot only positive frequencies
freq ,spec       = psd.rspec()
freq,qispec      = psd.rspec(qispec)

fig = plt.figure()
ax = fig.add_subplot(3,1,1)
ax.plot(t/3600,x)
ax = fig.add_subplot(3,2,3)
ax.loglog(freq,psd.sk[0:psd.nf,0])
ax.set_ylim(1e0,1e9)
ax.set_xlim(1e-5,1e0)
ax = fig.add_subplot(3,2,4)
ax.loglog(freq,spec)
ax.set_ylim(1e0,1e9)
ax.set_xlim(1e-5,1e0)
ax = fig.add_subplot(3,2,5)
ax.loglog(sine_psd.freq,sine_psd.spec)
ax.loglog(sine_psd.freq,sine_psd.err,'--')
ax.set_ylim(1e0,1e9)
ax.set_xlim(1e-5,1e0)
ax = fig.add_subplot(3,2,6)
ax.loglog(freq,qispec)
ax.set_ylim(1e0,1e9)
ax.set_xlim(1e-5,1e0)

plt.show()

