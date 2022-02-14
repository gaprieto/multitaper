# Testing DPSS codes. 

import multitaper.mtcross as cross
import numpy as np
import matplotlib.pyplot as plt
import multitaper.utils  as utils

def Qi(nf,trf,cohe,freq):

    Qi   = trf[1:nf]
    cohe = cohe[1:nf]
    wt   = np.zeros((nf-1),dtype=float)
    per  = freq[1:nf]
    per  = 1.0/per
    lper = np.log10(per)

    #-------------------------------------
    #  Compute Qi and wt
    #-------------------------------------

    for i in range(nf-1):
        if (cohe[i]>= 0.6):
            wt[i] = 1.0/np.sqrt(1.0-cohe[i])
        else:
            wt[i] = 0.0

    #-------------------------------------------
    #  Band averaging 
    #     (as Constable and Constable (2004)
    #-------------------------------------------

    avper = np.zeros(10,dtype=float)
    avper[0]  = 21330.0 
    avper[1]  = 41410.0 
    avper[2]  = 74400.0 
    avper[3]  = 185100.0 
    avper[4]  = 348000.0
    avper[5]  = 697800.0
    avper[6]  = 1428000.0
    avper[7]  = 2674000.0 
    avper[8]  = 4593000.0 
    avper[9]  = 11810000.0

    avper = np.log10(avper)

    travg = np.zeros(10,dtype=complex)
    for i in range(10):
        per2loc = np.abs(lper-avper[i])
        iloc1 = np.where(per2loc<=0.1)
        iloc  = np.array(iloc1[0])
        nloc  = len(iloc) 
        if (nloc<1):
            continue

        swt = 0.0
        for k in range(nloc):
           k2 = iloc[k]
           travg[i] = travg[i] + wt[k2]*Qi[k2]
           swt      = swt + wt[k2]
        travg[i] = travg[i]/swt

    cavg = 6378. * (1. - 2.*travg) / (2.*(1.+travg)) 

    return avper, cavg

#------------------------------------------------
# Define filename and path
#------------------------------------------------

fname = 'asc_akima.dat'
print('fname ', fname)

#------------------------------------------------
# Define desired parameters
#------------------------------------------------
nw    = 7.5
kspec = 12 

#------------------------------------------------
# Load the data
#------------------------------------------------

data = utils.get_data(fname)
t    = data[:,0]
x    = data[:,2]
y    = data[:,1]

npts = np.shape(x)[0]
dt   = t[1]-t[0] 
print('npts, dt ', npts,dt)

#------------------------------------------------
# Get MTSPEC
#     Get reshape spectrum with F test
#     Get confidence intervals
#     Get QI spectrum
#------------------------------------------------

print('----- Sine cross spectrum -------')
Sxy  = cross.SineCross(x,y,ntap=10,dt=dt)
avper, sine_cavg = Qi(Sxy.nf,Sxy.trf,Sxy.cohe,Sxy.freq)

print('----- MT cross spectrum -------')
Pxy  = cross.MTCross(x,y,nw,kspec,dt)
avper, cavg = Qi(Pxy.nf,Pxy.trf,Pxy.cohe,Pxy.freq)

fig = plt.figure(1,figsize=(10,8))
ax1 = fig.add_subplot(2,2,1)
ax1.plot(t/1e6,x)
ax1.set_ylim(-120, 20)
ax1.set_ylabel('Magnetic Field (nT)')
ax1.set_xlabel(r'Time ($10^6$ s)')
ax2 = fig.add_subplot(2,2,2)
ax2.plot(t/1e6,y)
ax2.set_ylim(-120, 20)
ax2.set_xlabel(r'Time ($10^6$ s)')
ax2.yaxis.tick_right()
ax3 = fig.add_subplot(2,2,(3,4))
ax3.plot(Pxy.freq[0:Pxy.nf]*86400,Pxy.cohe[0:Pxy.nf])
ax3.plot(Sxy.freq*86400,Sxy.cohe)
ax3.set_ylabel(r'Squared Coherence')
ax3.set_xlabel(r'Frequency (cycles / day)')

fig = plt.figure(2)
ax = fig.add_subplot()
ax.plot(avper,np.real(cavg),label='Real MT')
ax.plot(avper,np.imag(cavg),label='Imag MT')
ax.plot(avper,np.real(sine_cavg),'--',label='Real sineMT')
ax.plot(avper,np.imag(sine_cavg),'--',label='Imag sineMT')
ax.set_xlim(4., 7.5)
ax.legend()
ax.set_ylabel(r'C - admittance (km)')
ax.set_xlabel(r'Log period (s)')
#plt.ylim(-750, 1800)

plt.show()
