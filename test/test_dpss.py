# Testing DPSS codes. 

import specpy.mtspec as mtspec
import numpy as np
import matplotlib.pyplot as plt

npts  = 100000
nw    = 4.0
kspec = 7

dpss, v = mtspec.dpss(npts,nw)#,kspec)

plt.plot(dpss)
plt.show()


