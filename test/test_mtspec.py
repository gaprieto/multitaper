# Testing DPSS codes. 

import specpy.mtspec as mtspec
import numpy as np
import matplotlib.pyplot as plt

npts  = 100
nw    = 3.5
kspec = 6

x = np.random.randn(npts)
x = np.loadtxt('test2.dat')

#spec, freq, se, spec_ci, F, p = mtspec.mtspec(x,nw,kspec,iadapt=0)
#
#plt.semilogy(freq,spec)
#plt.semilogy(freq,spec_ci,'r--')
#plt.xlim(0, 0.5)
#plt.show()
#
#plt.plot(freq,p)
#plt.xlim(0, 0.5)
#plt.show()

psd = mtspec.mtspec1d(x,nw,kspec,nfft=npts,iadapt=0)
print('psd.nw     ', psd.nw)
print('psd.kspec  ', psd.kspec)
print('psd.iadapt ', psd.iadapt)
print('psd.lamb ', psd.lamb)

respec, spec_noline, yk, sline = psd.reshape(fcrit=0.99)

plt.figure(10)
plt.plot(psd.rfreq,psd.rspec)
plt.plot(psd.rfreq,psd.rqispec)

plt.figure()
plt.plot(psd.freq,psd.p)

plt.figure()
plt.semilogy(psd.freq,psd.F)

plt.figure()
plt.loglog(psd.freq,respec)
plt.xlim(0.005, 0.5)
#plt.ylim(1e6, 1e12)

plt.figure()
plt.loglog(psd.freq,spec_noline)
plt.xlim(0.005, 0.5)
#plt.ylim(1e1, 1e9)

plt.figure()
plt.plot(psd.vn)
plt.show()

