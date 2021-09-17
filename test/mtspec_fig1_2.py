# Testing DPSS codes. 

import specpy.mtspec as mtspec
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------
# Define desired parameters
#------------------------------------------------
nw    = 3.5
kspec = 5 

#------------------------------------------------
# Load the data
#------------------------------------------------

x    = np.loadtxt('../data/v22_174_series.dat')
npts = np.shape(x)[0]
dt   = 4930
t    = np.arange(npts)*dt
print('npts, dt ', npts,dt)

#------------------------------------------------
# Get MTSPEC
#     Get reshape spectrum with F test
#     Get confidence intervals
#     Get QI spectrum
#------------------------------------------------

psd = mtspec.mtspec(x,nw,kspec,dt,iadapt=0)

print(type(psd))

# Reshape spectrum
F,p = psd.ftest()
respec, spec_noline, yk, sline = psd.reshape(fcrit=0.90,p=p)
# Confidence intervals
spec_ci = psd.jackspec()
# QI inverse spectrum
qispec  = psd.qiinv()[0]

# Plot only positive frequencies
freq ,spec               = psd.rspec()
freq,qispec,spec_ci      = psd.rspec(qispec,spec_ci)
freq,respec,spec_noline  = psd.rspec(respec,spec_noline)
F = F[0:psd.nf]

plt.figure(1)
plt.plot(t/1000,x)

plt.figure(2)
plt.semilogy(freq*1e6,spec)
plt.semilogy(freq*1e6,spec_ci,'k--')
plt.semilogy(freq*1e6,qispec,'r--')
plt.xlim(0, 100)

plt.figure(3)
plt.plot(freq*1e6,F)
plt.xlim(0,100)
#plt.ylim(-0.5,14.5)

plt.figure(4)
plt.semilogy(freq*1e6,respec)
plt.xlim(0,100)

plt.figure(5)
plt.semilogy(freq*1e6,spec_noline)
plt.xlim(0,100)

plt.figure(6)
plt.semilogy(freq*1e6,qispec)
plt.xlim(0,100)

plt.show()

#plt.plot(psd.rfreq,psd.rspec)
#plt.plot(psd.rfreq,psd.rqispec)
#
#plt.figure()
#plt.plot(psd.freq,psd.p)
#
#plt.figure()
#plt.semilogy(psd.freq,psd.F)
#
#plt.figure()
#plt.loglog(psd.freq,respec)
#plt.xlim(0.005, 0.5)
#plt.ylim(1e6, 1e12)

#plt.figure()
#plt.loglog(psd.freq,spec_noline)
#plt.xlim(0.005, 0.5)
#plt.ylim(1e1, 1e9)

#plt.figure()
#plt.plot(psd.vn)
#plt.show()

