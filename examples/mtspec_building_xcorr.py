import specpy.mtspec as mtspec
import specpy.utils as utils
import specpy.mtcross as mtcross
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../data/crisanto_mesetas.dat')
dt   = 1/200.
npts,ntr = np.shape(data)

print('npts, # of traces, dt ',npts, ntr, dt)

# create time vector
qtime = 3*60+55   # in seconds
t    = np.arange(npts)*dt - qtime
dmax = np.amax(data)/2

#-------------------------------------
# Define windows for deconvolution
#-------------------------------------
ta_1 = -3.9*60
ta_2 = -0.1*60
tb_1 = 0*60
tb_2 = 2*60
tc_1 = 7*60
tc_2 = 12*60
iloc1 = np.where((t>=ta_1) & (t<=ta_2))[0]
iloc2 = np.where((t>=tb_1) & (t<=tb_2))[0]
iloc3 = np.where((t>=tc_1) & (t<=tc_2))[0]

fig = plt.figure()
ax  = fig.add_subplot()
for i in range(ntr):
    ax.plot(t/60,data[:,i]/dmax+(i*3)+2,'k')
ax.set_xlabel('Time since M6.0 earthquake (min)')
ax.set_ylabel('Floor')
ax.set_yticks([2, 5, 8, 11, 14]);
ax.plot((ta_1/60,ta_1/60),(1,16),'b--')
ax.plot((ta_2/60,ta_2/60),(1,16),'b--')
ax.plot((tb_1/60,tb_1/60),(1,16),'r--')
ax.plot((tb_2/60,tb_2/60),(1,16),'r--')
ax.plot((tc_1/60,tc_1/60),(1,16),'k--')
ax.plot((tc_2/60,tc_2/60),(1,16),'k--')

#------------------------------------------------
# Define desired parameters
#------------------------------------------------
nw      = 3.5
kspec   = 5 
ifloor1 = 0

#------------------------------------------------
# Define filter desired parameters
#------------------------------------------------

fmin = 0.5
fmax = 10.0
fnyq = 0.5/dt
wn   = [fmin/fnyq,fmax/fnyq]
b, a = signal.butter(4, wn,'bandpass')


print('----- MT IRF calculation -------')

x = data[iloc1,ifloor1]
fig = plt.figure()
ax  = fig.add_subplot()
for i in range(ntr):
    print(i)
    y = data[iloc1,i]
    Pxy  = mtcross.mtcross(y,x,nw,kspec,dt)
    xcorr, dcohe, dconv  = Pxy.mt_corr()
    dconv = signal.filtfilt(b, a, dcohe[:,0])

    k    = np.linspace(-Pxy.npts,Pxy.npts,len(xcorr),dtype=int)
    t2   = k*dt
    tloc = np.where((t2>=-2.0) & (t2<=2))[0]
    ax.plot(t2[tloc],dconv[tloc]/np.max(dconv[tloc])+i,'b')
    
print('----------------------------------')

print('----- MT IRF calculation -------')

x = data[iloc2,ifloor1]
for i in range(ntr):
    print(i)
    y = data[iloc2,i]
    Pxy  = mtcross.mtcross(y,x,nw,kspec,dt)
    xcorr, dcohe, dconv  = Pxy.mt_corr()
    dconv = signal.filtfilt(b, a, dcohe[:,0])

    k    = np.linspace(-Pxy.npts,Pxy.npts,len(xcorr),dtype=int)
    t2   = k*dt
    tloc = np.where((t2>=-2.0) & (t2<=2))[0]
    ax.plot(t2[tloc],dconv[tloc]/np.max(dconv[tloc])+i,'r')
    
print('----------------------------------')

print('----- MT IRF calculation -------')

x = data[iloc3,ifloor1]
for i in range(ntr):
    print(i)
    y = data[iloc3,i]
    Pxy  = mtcross.mtcross(y,x,nw,kspec,dt)
    xcorr, dcohe, dconv  = Pxy.mt_corr()
    dconv = signal.filtfilt(b, a, dcohe[:,0])

    k    = np.linspace(-Pxy.npts,Pxy.npts,len(xcorr),dtype=int)
    t2   = k*dt
    tloc = np.where((t2>=-2.0) & (t2<=2))[0]
    ax.plot(t2[tloc],dconv[tloc]/np.max(dconv[tloc])+i,'k')
    
print('----------------------------------')
ax.set_xlim(-2, 2)
plt.show()



