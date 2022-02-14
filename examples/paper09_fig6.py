# Testing DPSS codes. 

import multitaper.utils  as utils
import multitaper.mtcross as cross
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------
# Define desired parameters
#------------------------------------------------

dt    = 1
nw    = 4.0
kspec = 7

#------------------------------------------------
# Load the data
#------------------------------------------------

fname1 = 'PASC.dat'
fname2 = 'ADO.dat'

x    = utils.get_data(fname1)
y    = utils.get_data(fname2)

#------------------------------------------------
# Get MTSPEC
#     Get reshape spectrum with F test
#     Get confidence intervals
#     Get QI spectrum
#------------------------------------------------

print('----- Sine IRF calculation -------')
Sxy      = cross.SineCross(x,y,ntap=20,dt=dt)
Sxc, Sch, Sdcnv  = Sxy.mt_corr()
Si = np.linspace(Sxy.nf-500,Sxy.nf+500,1001,dtype=int)
St = Si-float(Sxy.nf)
print('----------------------------------')

print('----- MT IRF calculation -------')
Pxy  = cross.MTCross(x,y,nw,kspec,dt)
xcorr, dcohe, dconv  = Pxy.mt_corr()
i = np.linspace(Pxy.npts-500,Pxy.npts+500,1001,dtype=int)
t = i-float(Pxy.npts)
print('----------------------------------')

fig = plt.figure(1,figsize=(10,8))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(-t,xcorr[i]/np.max(np.abs(xcorr[i])))
ax1.plot(-t,dcohe[i]/np.max(np.abs(dcohe[i]))+2.0)
ax1.plot(-t,dconv[i]/np.max(np.abs(dconv[i]))+4.0)
ax1.set_title('Multitaper deconvolution')
ax1.text(200,-0.3,'cross-correlation')
ax1.text(200,2.0-0.3,'coherency')
ax1.text(200,4.0-0.3,'deconvolution')
ax1.set_xlim(0,500)

ax2 = fig.add_subplot(2,1,2)
ax2.plot(-St,Sxc[Si]/np.max(np.abs(Sxc[Si]))+0.0)
ax2.plot(-St,Sch[Si]/np.max(np.abs(Sch[Si]))+2.0)
ax2.plot(-St,Sdcnv[Si]/np.max(np.abs(Sdcnv[Si]))+4.0)
ax2.set_title('Sine deconvolution')
ax2.text(200,-0.3,'cross-correlation')
ax2.text(200,2.0-0.3,'coherency')
ax2.text(200,4.0-0.3,'deconvolution')
ax2.set_xlim(0,500)


plt.show()


