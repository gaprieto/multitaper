# Testing DPSS codes. 

import multitaper.mtspec as mtspec
import multitaper.utils as utils 
import numpy as np
import matplotlib.pyplot as plt

npts  = 100
nw    = 4.0
kspec = 7

dpss, v   = utils.dpss2(npts,nw,kspec)
dpss1, v1 = utils.dpss(npts,nw,kspec)

print(v, v1)
plt.figure()
plt.plot(dpss[:,0],'k')
plt.plot(dpss[:,3],'k')
plt.plot(dpss[:,6],'k')
#plt.plot(dpss1,'r--')


plt.show()


