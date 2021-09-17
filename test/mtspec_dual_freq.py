# Testing DPSS codes. 

import specpy.mtspec as mtspec
import specpy.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm

#------------------------------------------------
# Define desired parameters
#------------------------------------------------
nw    = 4.0
kspec = 6 

#------------------------------------------------
# Load the data
#------------------------------------------------

x    = np.loadtxt('../data/pasc.dat')
y    = np.loadtxt('../data/ado.dat')
npts = np.shape(x)[0]
dt   = 1.0
t    = np.arange(npts)*dt
print('npts, dt ', npts,dt)

#------------------------------------------------
# Get MTSPEC
#     Get reshape spectrum with F test
#     Get confidence intervals
#     Get QI spectrum
#------------------------------------------------

deg2rad = np.pi/180.
nwin = 600
i = np.linspace(0,npts-nwin,6*24)
df_cnt = 0
for j in range(1000):
   i1 = j*nwin
   i2 = i1+nwin
   pdone = float(i2)/float(npts)*100.0
   if (np.abs(pdone/5-np.round(pdone/5)) < 0.1):
      print('Percent done %5.1f ' %(pdone))
   if (i2 > npts):
      break
 
   px = mtspec.mtspec(x[i1:i2],nw,kspec,dt)
   py = mtspec.mtspec(y[i1:i2],nw,kspec,dt)
   df_spec, df_cohe, df_phase, freq = utils.df_spec(px,py,fmax=0.2)
   df_Spx, df_cpx, df_phpx, freq = utils.df_spec(px,fmax=0.2)
   df_Spy, df_cpy, df_phpy, freq = utils.df_spec(py,fmax=0.2)
   nf2 = np.shape(df_spec)[0]
   if (j==0):
      ph      = np.zeros((nf2,nf2,2))
      ph[:,:,0] = np.cos(df_phase)
      ph[:,:,1] = np.sin(df_phase)   
#      ph_px = np.ndarray((np.cos(df_phpx),np.sin(df_phpx)))
#      ph_py = np.ndarray((np.cos(df_phpy),np.sin(df_phpy)))
      df    = df_cohe   #[1:nf,1:nf]
      df_px = df_cpx
      df_py = df_cpy
   else:
#      ph    = ph    + np.ndarray((np.cos(df_phase),np.sin(df_phase)))
      ph[:,:,0] = ph[:,:,0] + np.cos(df_phase)
      ph[:,:,1] = ph[:,:,1] + np.sin(df_phase)   
#      ph_px = ph_px + np.ndarray((np.cos(df_phpx),np.sin(df_phpx)))
#      ph_py = ph_py +  np.ndarray((np.cos(df_phpy),np.sin(df_phpy)))
      df    = df    + df_cohe  #[1:nf,1:nf]
      df_px = df_px + df_cpx
      df_py = df_py + df_cpy

   df_cnt = df_cnt + 1

df    = df/df_cnt
df_px = df_px/df_cnt
df_py = df_py/df_cnt
ph    = ph/df_cnt
ph    = np.arctan2(ph[:,:,1],ph[:,:,0])   
#ph_px = ph_px/df_cnt
#ph_py = ph_py/df_cnt
df_px = np.minimum(df_px,0.3)
df_py = np.minimum(df_py,0.3)
df    = np.minimum(df,0.3)
#freq = px.freq[1:nf]

X, Y = np.meshgrid(freq,freq)

fig = plt.figure(1)
plt.figure(1)
ax1 = fig.add_subplot(2,1,1)
#c = ax1.pcolor(X, Y, df,
#               norm=cm.LogNorm(), cmap='Greys', shading='auto')
c = ax1.pcolor(X, Y, df,
               cmap='Greys', shading='auto')
ax1.get_xaxis().set_ticks([])
fig.colorbar(c, ax=ax1)
ax1.set_xlim(0, 0.2)
ax1.set_ylim(0, 0.2)
ax1.set_title('PASC-ADO Coherence')

ax2 = fig.add_subplot(2,1,2)
c = ax2.pcolor(X, Y, ph,
               cmap='Greys', shading='auto')
ax2.get_xaxis().set_ticks([])
fig.colorbar(c, ax=ax2)
ax2.set_xlim(0, 0.2)
ax2.set_ylim(0, 0.2)
ax2.set_title('PASC-ADO Phase')
plt.savefig('pasc_ado.png')


fig = plt.figure(2)
plt.figure(1)
ax1 = fig.add_subplot(2,1,1)
c = ax1.pcolor(X, Y, df_px,
               cmap='Greys', shading='auto')
ax1.get_xaxis().set_ticks([])
fig.colorbar(c, ax=ax1)
ax1.set_xlim(0, 0.2)
ax1.set_ylim(0, 0.2)
ax1.set_title('PASC Auto-Coherence')

ax2 = fig.add_subplot(2,1,2)
c = ax2.pcolor(X, Y, ph, #_px,
               cmap='Greys', shading='auto')
ax2.get_xaxis().set_ticks([])
fig.colorbar(c, ax=ax2)
ax2.set_xlim(0, 0.2)
ax2.set_ylim(0, 0.2)
ax2.set_title('PASC Auto-Phase')
plt.savefig('pasc.png')

fig = plt.figure(3)
plt.figure(1)
ax1 = fig.add_subplot(2,1,1)
c = ax1.pcolor(X, Y, df_py,
               cmap='Greys', shading='auto')
#c = ax1.pcolor(X, Y, df_py,
#               norm=cm.LogNorm(), cmap='Greys', shading='auto')
ax1.get_xaxis().set_ticks([])
fig.colorbar(c, ax=ax1)
ax1.set_xlim(0, 0.2)
ax1.set_ylim(0, 0.2)
ax1.set_title('ADO Auto-Coherence')

ax2 = fig.add_subplot(2,1,2)
c = ax2.pcolor(X, Y, ph, #_py,
               cmap='Greys', shading='auto')
ax2.get_xaxis().set_ticks([])
fig.colorbar(c, ax=ax2)
ax2.set_xlim(0, 0.2)
ax2.set_ylim(0, 0.2)
ax2.set_title('ADO Auto-Phase')
plt.savefig('ado.png')

plt.show()


