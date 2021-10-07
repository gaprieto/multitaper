import specpy.mtspec as mtspec
import specpy.utils as utils
import specpy.mtcross as mtcross
import numpy as np
import matplotlib.pyplot as plt

dt   = 1.0
npts = 100
t    = np.arange(npts)*dt 

#data = np.random.randn(npts)
#x    = (2.*np.cos(2*np.pi*t*0.1) + 
#       1.*np.sin(2*np.pi*t*0.25) + data)

#np.savetxt('data1.dat',x) 

# simple test
#x = np.loadtxt('data1.dat')

# new data
x2 = np.loadtxt('crisanto_mesetas.dat')
x    = x2[0:4000,4]
npts = len(x)
dt   = 1/200.
#------------------------------------------------
# Define desired parameters
#------------------------------------------------
nw     = 3.5
kspec  = 5 

vn,theta = utils.dpss(npts,nw,kspec)
psd = mtspec.mtspec(x,nw,kspec,dt,iadapt=0,
                        vn=vn,lamb=theta)

print(psd.nf)
nf = psd.nf
freq    = psd.freq[0:nf,0]
spec    = psd.spec[0:nf,0] 
qispec  = psd.qiinv()[0]
qispec  = qispec[0:nf]
   
print('Min(qispec) ',np.min(qispec))
 

fig = plt.figure()
ax1 = fig.add_subplot(2,1,2)
ax1.plot(freq,spec);
ax1.plot(freq,qispec);
ax2 = fig.add_subplot(2,1,1)
ax2.plot(x)
plt.show()
