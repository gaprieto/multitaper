#---------------------
# Load modules needed
#---------------------

import specpy.mtspec as mtspec
import specpy.utils as utils
import specpy.mtcross as mtcross
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import obspy
from obspy.core.utcdatetime import UTCDateTime
import glob

def df_win(x,y=None,dt=None,fname=None,nwin=300):

    #------------------------------------------------
    # Define desired parameters
    #------------------------------------------------
    nw     = 4.0
    kspec  = 6 
    fmin   = 0.01
    fmax   = 0.2
    nperc  = 10 
    print(fname)

    npts = np.shape(x)[0]
    i = np.linspace(0,npts-nwin,npts)

    df_cnt = 0
    for j in range(1000):#1000):
        i1 = j*int(nwin/4)
        i2 = i1+nwin
        pdone = float(i2)/float(npts)*100.0
        if (int(pdone)%nperc==0):
            print('Percent done %5.1f ' %(pdone))
        if (i2 > npts):
            break 

        if (y is None):
            data = x[i1:i2]
            t    = np.linspace(0,npts,npts)*dt
            px  = mtspec.mtspec(data,nw,kspec,dt)
            df_Spx, df_cpx, df_phpx, freq = utils.df_spec(px,fmin=fmin,fmax=fmax)
        else:
            data  = x[i1:i2]
            data2 = y[i1:i2]
            t    = np.linspace(0,npts,npts)*dt
            px  = mtspec.mtspec(data,nw,kspec,dt)
            py  = mtspec.mtspec(data2,nw,kspec,dt)
            df_Spx, df_cpx, df_phpx, freq = utils.df_spec(px,py,fmin=fmin,fmax=fmax)

        df_cnt = df_cnt + 1
        if (df_cnt == 1):
            fig = plt.figure(figsize=(10,10))
            ax0 = fig.add_subplot(5,1,df_cnt)
            if (y is None):
                ax0.plot(t,x/max(np.abs(x)))
                ax0.set_ylim(-1.2, 1.2)
            else:
                ax0.plot(t,x/max(np.abs(x)))
                ax0.plot(t,y/max(np.abs(y))+2)
            ax0.get_xaxis().set_ticks([])
            ax0.get_yaxis().set_ticks([])


        ax0.text(t[i1],1.0,df_cnt)
        X, Y = np.meshgrid(freq,freq)
#        X, Y = np.meshgrid(np.log10(freq),np.log10(freq))
#        ax0 = fig.add_subplot(4,1,1)
#        ax0.plot(t,x)
#        ax0.plot(t[i1:i2],data)
        ax1 = fig.add_subplot(6,5,df_cnt+5)
        c = ax1.pcolor(X, Y, df_cpx,
                       cmap='Greys', shading='auto',
                       vmin=0.0, vmax=1.0)
#        fig.colorbar(c, ax=ax1)
        ax1.set_aspect('equal', 'box')
        if (df_cnt<19):
            ax1.get_xaxis().set_ticks([])
        if ((df_cnt+4)%5>0):
            ax1.get_yaxis().set_ticks([])

        if (df_cnt == 25):
            df_cnt = 0


    plt.savefig(fname)
    plt.close()
    # plt.show()
#---------------
# Main code
#---------------


evid = '02'

fld    = "wv/"
ffld   = 'figs/'+evid+'_'
ffind  = fld+evid+'_CM*HHZ*.mseed'
S1     = glob.glob(ffind)
twin   = 600.   # 10 min windows

S1.sort()
nfile = len(S1) 

for f1 in S1:
    
    # Read file
    st      = obspy.read(f1)
    ntrace  = len(st)
    if (ntrace!=1):
        print('multiple traves')
        print(st)
        continue
    #pre_filt = [0.001, 0.005, 45, 50]
    #st.remove_response(inventory=inv,output="VEL",
    #                   pre_filt=pre_filt,
    #                   water_level=60)

    dt      = st[0].stats.delta
    sta     = st[0].stats.station
    chan    = st[0].stats.channel
    if (dt==0.05):
       st      = st.decimate(5) 
       st      = st.decimate(4)
    elif(dt==0.01):
       st      = st.decimate(10)
    elif(dt==0.005):
       st      = st.decimate(5)
       st      = st.decimate(4)
    ntrace  = len(st)
    if (ntrace==0):
        continue
    wv      = st[0].data
    npts    = st[0].stats.npts
    dt      = st[0].stats.delta

    #st.plot()
    fname = ffld+sta+'_'+chan+'.png'
    nwin = int(twin/dt)
    print(ntrace, npts, nwin, dt, sta, chan)

    df_win(wv,dt=dt,fname=fname,nwin=nwin)


#---------------------------------------
# Now the cross-coherence
#---------------------------------------

for i_file in range(nfile-1):
    break
    f1 = S1[i_file]
    # Read file
    st      = obspy.read(f1)
    ntrace  = len(st)
    if (ntrace==0):
        continue
    dt      = st[0].stats.delta
    sta     = st[0].stats.station
    chan    = st[0].stats.channel
    if (dt==0.05):
       st      = st.decimate(5) 
       st      = st.decimate(4)
    elif(dt==0.01):
       st      = st.decimate(10)
    elif(dt==0.005):
       st      = st.decimate(5)
       st      = st.decimate(4)
    ntrace  = len(st)
    if (ntrace==0):
        continue
    wv      = st[0].data
    npts    = st[0].stats.npts
    dt      = st[0].stats.delta
    for j_file in range(i_file+1,nfile):
        f2 = S1[j_file]
        # Read file
        st2      = obspy.read(f2)
        ntrace   = len(st2)
        if (ntrace==0):
            continue
        dt2      = st2[0].stats.delta
        sta2     = st2[0].stats.station
        chan2    = st2[0].stats.channel

        if (sta==sta2):
            continue

        if (dt2==0.05):
            st2      = st2.decimate(5) 
            st2      = st2.decimate(4)
        elif(dt2==0.01):
            st2      = st2.decimate(10)
        elif(dt2==0.005):
            st2      = st2.decimate(5)
            st2      = st2.decimate(4)
        ntrace2  = len(st2)
        if (ntrace2==0):
            continue
        wv2      = st2[0].data
        npts2    = st2[0].stats.npts
        dt2      = st2[0].stats.delta

        #st.plot()
        fname = ffld+sta+'_'+chan+'_'+sta2+'_'+chan2+'.png'
        nwin = int(twin/dt)
        print(ntrace, npts, nwin, dt, sta, chan,sta2,chan2)

        df_win(wv,wv2,dt,fname,nwin=nwin)


   




