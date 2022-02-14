# Testing DPSS codes. 

import multitaper.utils  as utils
import multitaper.mtspec as spec
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------
# Define filename and path
#------------------------------------------------

fname = 'v22_174_series.dat'
print('fname ', fname)

#------------------------------------------------
# Define desired parameters
#------------------------------------------------
nw    = 3.5
kspec = 5 

#------------------------------------------------
# Load the data
#------------------------------------------------

x    = utils.get_data(fname)
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

psd = spec.MTSpec(x,nw,kspec,dt,iadapt=0)

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

fig = plt.figure(1)
ax  = fig.add_subplot()
ax.plot(t/1000,x)
ax.set_xlabel('time (kyear)')
ax.set_ylabel(r'Change in $\delta^{18}O$')

fig = plt.figure(2)
ax  = fig.add_subplot()
ax.semilogy(freq*1e6,spec)
ax.semilogy(freq*1e6,spec_ci,'k--')
ax.semilogy(freq*1e6,qispec,'r--')
ax.set_title('Spectrum estimate with 95% confidence interval')
ax.set_xlim(0, 100)
ax.set_xlabel(r'Frequency ($c Ma^{-1}$)')
ax.set_ylabel(r'Power spectral density ($\delta^{18}O/ca^{-1}$)')

fig = plt.figure(3)
ax  = fig.add_subplot()
ax.plot(freq*1e6,F)
ax.set_xlim(0,100)
ax.set_title(r'$F$ statistic for periodic components')
ax.set_xlabel(r'Frequency ($c Ma^{-1}$)')
ax.set_ylabel(r'F-statistic for periodic lines')
#plt.ylim(-0.5,14.5)

fig = plt.figure(4)
ax  = fig.add_subplot()
ax.semilogy(freq*1e6,respec)
ax.set_xlim(0,100)
ax.set_title(r'Reshaped spectrum')
ax.set_xlabel(r'Frequency ($c Ma^{-1}$)')
ax.set_ylabel(r'Power spectral density ($\delta^{18}O/ca^{-1}$)')
#plt.ylim(-0.5,14.5)

plt.figure(5)
plt.semilogy(freq*1e6,spec_noline)
plt.xlim(0,100)

plt.figure(6)
plt.semilogy(freq*1e6,qispec)
plt.xlim(0,100)

plt.show()
