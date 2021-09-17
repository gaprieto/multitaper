# Testing Dual-frequency codes

import specpy.mtspec as mtspec
import specpy.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm

#------------------------------------------------
# Define desired parameters
#------------------------------------------------
nw    = 6.5
kspec = 10 

#------------------------------------------------
# Load the data
#------------------------------------------------

npts = 601
dt   = 1.0
t    = np.arange(npts)*dt

mu, sigma = 0.0, 10.0
x_noise  = np.random.normal(mu,sigma,npts)
x_noise2 = np.random.normal(mu,sigma,npts)

fa  = 0.030
fb  = 0.075
fc  = 0.050
f_t  = (fb-fa)*t/np.max(t) + fa 
f_t2 = (fc-fa)*t/np.max(t) + fa 
print(np.min(f_t),np.max(f_t))
#fx = 100.0*np.sin(2*np.pi*t*fb)+x_noise
fx = 100.0*np.sin(2*np.pi*t*f_t) + x_noise
fy = 100.0*np.sin(2*np.pi*t*f_t2)+ x_noise2
fz = (50.0*np.sin(2*np.pi*t*f_t) + 
      +  50.0*np.sin(2.5*2*np.pi*t*f_t)) + x_noise

print('npts, dt ', npts,dt)
print(np.shape(fx))

#------------------------------------------------
# Get MTSPEC
#------------------------------------------------

P_noise = mtspec.mtspec(x_noise, nw, kspec, dt)
Px      = mtspec.mtspec(fx, nw, kspec, dt)
Py      = mtspec.mtspec(fy, nw, kspec, dt)
Pz      = mtspec.mtspec(fz, nw, kspec, dt)

df_nspec, df_nch, df_nph, freqn = utils.df_spec(P_noise, fmax=0.18)
df_spec,  df_ch,  df_ph,  freq  = utils.df_spec(Px, fmax=0.18)
xy_spec,  xy_ch,  xy_ph,  freq  = utils.df_spec(Px,Py, fmax=0.18)
z_spec,   z_ch,   z_ph,   freq  = utils.df_spec(Pz, fmax=0.18)
xz_spec,  xz_ch,  xz_ph,  freq  = utils.df_spec(Px,Pz, fmax=0.18)
nf  = Px.nf

#--------------------------------------------------
# Plot results
#--------------------------------------------------

fig = plt.figure(1)
ax1 = fig.add_subplot(2,1,1)
ax1.plot(t,x_noise)
ax2 = fig.add_subplot(2,1,2)
ax2.plot(t,fx)

fig = plt.figure(2)
ax1 = fig.add_subplot(2,1,1)
ax1.plot(P_noise.freq[0:nf],P_noise.spec[0:nf])
ax2 = fig.add_subplot(2,1,2)
ax2.semilogy(Px.freq[0:nf],Px.spec[0:nf])
ax1.set_xlim(0, 0.18)
ax2.set_xlim(0, 0.18)

X, Y = np.meshgrid(freq,freq)

fig = plt.figure(3)
ax1 = fig.add_subplot(2,2,1)
c = ax1.pcolormesh(X, Y, df_nch,
                cmap='Greys', shading='auto',vmin=-0.5)
fig.colorbar(c, ax=ax1)
ax1.set_aspect('equal', 'box')
ax1.set_xlim(0, 0.18)
ax1.set_ylim(0, 0.18)
#ax1.set_clim(-0.5,1.0)

ax2 = fig.add_subplot(2,2,2)
c = ax2.pcolormesh(X, Y, df_ch,
                cmap='Greys', shading='auto',vmin=-0.5)
fig.colorbar(c, ax=ax2)
ax2.set_aspect('equal', 'box')
ax2.set_xlim(0, 0.18)
ax2.set_ylim(0, 0.18)

ax3 = fig.add_subplot(2,2,3)
c = ax3.pcolor(X, Y, df_nph,
                cmap='Greys', shading='auto')
fig.colorbar(c, ax=ax3)
ax3.set_aspect('equal', 'box')
ax3.set_xlim(0, 0.18)
ax3.set_ylim(0, 0.18)
ax4 = fig.add_subplot(2,2,4)
c = ax4.pcolor(X, Y, df_ph,
                cmap='Greys', shading='auto')
fig.colorbar(c, ax=ax4)
ax4.set_aspect('equal', 'box')
ax4.set_xlim(0, 0.18)
ax4.set_ylim(0, 0.18)

fig = plt.figure(4)
ax1 = fig.add_subplot(2,1,1)
c = ax1.pcolormesh(X, Y, xy_ch,
                cmap='Greys', shading='auto',vmin=-0.5)
fig.colorbar(c, ax=ax1)
ax1.set_aspect('equal', 'box')
ax1.set_xlim(0, 0.18)
ax1.set_ylim(0, 0.18)
ax2 = fig.add_subplot(2,1,2)
ax2.semilogy(Px.freq[0:nf],Px.spec[0:nf])
ax2.semilogy(Py.freq[0:nf],Py.spec[0:nf])
ax2.set_xlim(0, 0.18)

fig = plt.figure(5)
ax1 = fig.add_subplot(2,1,1)
c = ax1.pcolormesh(X, Y, z_ch,
                cmap='Greys', shading='auto',vmin=-0.5)
fig.colorbar(c, ax=ax1)
ax1.set_aspect('equal', 'box')
ax1.set_xlim(0, 0.18)
ax1.set_ylim(0, 0.18)
ax2 = fig.add_subplot(2,1,2)
ax2.semilogy(Px.freq[0:nf],Pz.spec[0:nf])
ax2.set_xlim(0, 0.18)


fig = plt.figure(6)
ax1 = fig.add_subplot(2,1,1)
c = ax1.pcolormesh(X, Y, xz_ch,
                cmap='Greys', shading='auto',vmin=-0.5)
fig.colorbar(c, ax=ax1)
ax1.set_aspect('equal', 'box')
ax1.set_xlim(0, 0.18)
ax1.set_ylim(0, 0.18)
ax2 = fig.add_subplot(2,1,2)
ax2.semilogy(Px.freq[0:nf],Px.spec[0:nf])
ax2.semilogy(Pz.freq[0:nf],Pz.spec[0:nf])
ax2.set_xlim(0, 0.18)

plt.show()


