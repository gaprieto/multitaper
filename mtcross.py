"""
Module that contains all multivariate-multitaper codes. 
Contains:
   mt_cohe
   mt_deconv
   TO DO 
      df_spec
      wv_spec
"""

#-----------------------------------------------------
# Import main libraries and modules
#-----------------------------------------------------

import numpy as np
import scipy
from scipy import signal
import scipy.linalg      as linalg
import matplotlib.pyplot as plt
import multitaper.utils      as utils 
import multitaper.mtspec     as mtspec

class mtcross:

    def __init__(self,x,y,nw=4,kspec=0,dt=1.0,nfft=0,iadapt=0,wl=0.0):
        """
        put some notes here. 
        Driver for multivariate MT spectrum analysis
        """
        
        #-----------------------------------------------------
        # Check if input data is MTSPEC class
        #-----------------------------------------------------

        if (type(x) is not type(y)):
            raise ValueError("X and Y are not similar types")

        if (type(x) is np.ndarray):
            
            #-----------------------------------------------------
            # Check dimensions of input vectors
            #-----------------------------------------------------

            xdim  = x.ndim
            ydim  = y.ndim
            if (xdim>2 or ydim>2):
                raise ValueError("Arrays cannot by 3D")
            if (xdim==1):
                x = x[:, np.newaxis]
            if (ydim==1):
                y = y[:, np.newaxis]
            if (x.shape[0] != y.shape[0]):
                raise ValueError('Size of arrays must be the same')
            ndim = x.ndim 
            nx   = x.shape[1]
            ny   = y.shape[1]
            npts = x.shape[0]
            if (nx>1 or ny>1):
                raise ValueError("Arrays must be a single column")

            x = mtspec.mtspec(x,nw,kspec,dt,nfft,iadapt=iadapt)
            y = mtspec.mtspec(y,nw,kspec,dt,nfft,iadapt=iadapt,vn=x.vn,lamb=x.lamb)

        #------------------------------------------------------------
        # Now, check MTSPEC variables have same sizes
        #------------------------------------------------------------

        if (x.npts != y.npts):
            raise ValueError("npts must coincide")
        if (x.dt != y.dt):
            raise ValueError("dt must coincide")
        if (x.nfft != y.nfft):
            raise ValueError("nfft must coincide")
        if (x.nw != y.nw):
            raise ValueError("NW must coincide")
        if (x.kspec != y.kspec):
            raise ValueError("KSPEC must coincide")

        #------------------------------------------------------------
        # Parameters based on MTSPEC class, not on input
        #------------------------------------------------------------
 
        iadapt = x.iadapt
        dt     = x.dt
        kspec  = x.kspec
        nfft   = x.nfft
        npts   = x.npts
        nw     = x.nw

        #------------------------------------------------------------
        # Create the cross and auto spectra
        #------------------------------------------------------------

        wt = np.minimum(x.wt,y.wt)
        se = utils.wt2dof(wt)

        wt_scale = np.sum(np.abs(wt)**2, axis=1)  # Scale weights to keep power 
        for k in range(kspec):
            wt[:,k] = wt[:,k]/np.sqrt(wt_scale)

        # Weighted Yk's
        dyk_x = np.zeros((nfft,kspec),dtype=complex)
        dyk_y = np.zeros((nfft,kspec),dtype=complex)
        for k in range(kspec):
            dyk_x[:,k] = wt[:,k] * x.yk[:,k]
            dyk_y[:,k] = wt[:,k] * y.yk[:,k]

        # Auto and Cross spectrum
        Sxy      = np.zeros((nfft,1),dtype=complex)
        Sxx      = np.zeros((nfft,1),dtype=float)
        Syy      = np.zeros((nfft,1),dtype=float)
        Sxx[:,0] = np.sum(np.abs(dyk_x)**2, axis=1) 
        Syy[:,0] = np.sum(np.abs(dyk_y)**2, axis=1) 
        Sxy[:,0] = np.sum(dyk_x * np.conjugate(dyk_y),axis=1)

        # Get coherence and phase
        cohe  = np.zeros((nfft,1),dtype=float)
        cohy  = np.zeros((nfft,1),dtype=complex)
        trf   = np.zeros((nfft,1),dtype=complex)
        phase = np.zeros((nfft,1),dtype=float)
        
        w_lev = wl*np.mean(Syy[:,0])
        #print(np.mean(Syy[:,0]),w_lev)
        for i in range(nfft):
            phase[i,0] = np.arctan2(np.imag(Sxy[i,0]),np.real(Sxy[i,0])) 
            cohe[i,0]  = np.abs(Sxy[i,0])**2 / (Sxx[i,0]*Syy[i,0])
            cohy[i,0]  = Sxy[i,0] / np.sqrt(Sxx[i,0]*Syy[i,0])
            trf[i,0]   = Sxy[i,0] / (Syy[i,0]+w_lev)

        phase = phase * (180.0/np.pi)

        #-----------------------------------------------------------------
        # Save all variables in self
        #-----------------------------------------------------------------

        self.freq   = x.freq
        self.dt     = dt
        self.df     = x.df
        self.nf     = x.nf
        self.nw     = nw
        self.kspec  = kspec
        self.nfft   = nfft
        self.npts   = npts
        self.iadapt = iadapt

        self.Sxx    = Sxx
        self.Syy    = Syy
        self.Sxy    = Sxy
        self.cohe   = cohe
        self.cohy   = cohy
        self.trf    = trf
        self.phase  = phase
        self.se     = se

        del Sxx, Syy, Sxy, cohe, phase, se

    #-------------------------------------------------------------------------
    # Finished INIT mvspec
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Deconvolution
    # TF = Sx/Sy 
    #    although actually we compute Sx*conj(Sy)/(Sy^2)
    # Take the IFFT to convert to the time domain. 
    # Assumes a real deconvolved signal (real input signals). 
    #-------------------------------------------------------------------------

    def mt_deconv(self): 

        """
        Compute the transfer function between two signals (cross spetrum
        already pre-computed
        """

        nfft  = self.nfft
        trf   = self.trf

        dfun  = scipy.fft.ifft(trf[:,0],nfft) 
        dfun  = np.real(scipy.fft.fftshift(dfun))
        dfun  = dfun[:,np.nexaxis]
        dfun  = dfun/float(nfft) 

        return dfun 

    #----------------------------------------------------------------
    # The three correlation-based estimates in the time domain
    #    correlation (cross-spectrum)
    #    deconvolution (transfer function)
    #    norm correlation (coherency)
    # Correlation:
    #    Sxy = Sx*conj(Sy)
    # Deconvolution:
    #    Sxy/Sy = Sx*conj(Sy)/Sy^2
    # Coherency
    #    Sxy/sqrt(Sx*Sy)
    #
    #----------------------------------------------------------------

    def mt_corr(self): 

        """
        Compute the time domain transfer function between two signals (cross spetrum
        already pre-computed
        """

        nfft = self.nfft
        cohy = self.cohy
        trf  = self.trf
        xc   = self.Sxy

        xcorr  = scipy.fft.ifft(xc[:,0],nfft) 
        xcorr  = np.real(scipy.fft.fftshift(xcorr))
        xcorr  = xcorr[:,np.newaxis]
        xcorr  = xcorr/float(nfft) 

        dcohy  = scipy.fft.ifft(cohy[:,0],nfft) 
        dcohy  = np.real(scipy.fft.fftshift(dcohy))
        dcohy  = dcohy[:,np.newaxis]
        dcohy  = dcohy/float(nfft) 

        dconv  = scipy.fft.ifft(trf[:,0],nfft) 
        dconv  = np.real(scipy.fft.fftshift(dconv))
        dconv  = dconv[:,np.newaxis]
        dconv  = dconv/float(nfft) 

        return xcorr, dcohy, dconv

class sinecross:

    def __init__(self,x,y,ntap=0,ntimes=0,fact=1.0,dt=1.0,p=0.95):

        """
        put some notes here. 
        Driver for multivariate SINE Cross spectrum analysis
        """
        
        #-----------------------------------------------------
        # Check if input data is MTSPEC class
        #-----------------------------------------------------

        if (type(x) is not type(y)):
            raise ValueError("X and Y are not similar types")

        if (type(x) is np.ndarray):
            
            #-----------------------------------------------------
            # Check dimensions of input vectors
            #-----------------------------------------------------

            xdim  = x.ndim
            ydim  = y.ndim
            if (xdim>2 or ydim>2):
                raise ValueError("Arrays cannot by 3D")
            if (xdim==1):
                x = x[:, np.newaxis]
            if (ydim==1):
                y = y[:, np.newaxis]
            if (x.shape[0] != y.shape[0]):
                raise ValueError('Size of arrays must be the same')
            ndim = x.ndim 
            nx   = x.shape[1]
            ny   = y.shape[1]
            npts = x.shape[0]
            if (nx>1 or ny>1):
                raise ValueError("Arrays must be a single column")

        #------------------------------------------------------------
        # Set defaults
        #------------------------------------------------------------

        if (ntap < 2): 
            ntap = 0
        if (ntimes <= 0):
            ntimes = 2 
        if (fact <= 0.):
            fact = 1.
        if (p >= 1.0 or p<=0.0):
            p = 0.95

        #-----------------------------------------------------------------
        # Detrend, get variance
        #-----------------------------------------------------------------

        x     = signal.detrend(x,axis=0,type='constant')
        xvar  = np.var(x)
        y     = signal.detrend(y,axis=0,type='constant')
        yvar  = np.var(y)

        #-----------------------------------------------------------------
        # Define other parameters (nfft, nf, freq vector)
        #-----------------------------------------------------------------
     
        if (npts%2==0): 
            nf     = int(npts/2+1)
        else:
            nf     = int((npts+1)/2) 
        nfft       = np.int(2*npts)
        freq       = scipy.fft.rfftfreq(npts,dt)
        df         = freq[2]-freq[1]
        freq       = freq[:, np.newaxis]

        #------------------------------------------------------
        # Put main parameters in self
        #------------------------------------------------------

        self.x      = x
        self.xvar   = xvar
        self.x      = y
        self.xvar   = yvar
        self.freq   = freq
        self.dt     = dt
        self.df     = df
        self.nf     = nf
        self.ntap   = ntap
        self.nfft   = nfft
        self.npts   = npts
        self.ntimes = ntimes

        #-----------------------------------------------------
        # Get the FFT of the two time series
        #    Only one FFT is required
        #-----------------------------------------------------

        xy  = np.hstack((x,y))
        fx  = scipy.fft.fft(xy,axis=0,n=nfft,workers=2)

        #-----------------------------------------------------
        #  Check if constant tapers or adaptive method
        #-----------------------------------------------------

        if (ntap>0):

            #  Estimate uniform taper PSD
            #print('Uniform cohe(f) with ',ntap, ' tapers')
            #print('Time-bandwidth product ',0.5*ntap)

            sxy, ktap = utils.squick2(nfft,fx,nf,ntap)

        else:

            itap = int(3.0 + np.sqrt(fact*float(npts))/5.0)
            sxy, ktap = utils.sadapt2(nfft,fx,nf,df,itap,
                               ntimes,fact)

        #-----------------------------------------------------------
        # Normalize by variance if X data
        #-----------------------------------------------------------

        sscal = 0.5*(sxy[0,0]+sxy[nf-1,0])+np.sum(sxy[1:nf-1,0])
        sscal = xvar/(sscal*df)
        sxy   = sscal*sxy

        #------------------------------------------------------------
        # Degrees of freedom
        #------------------------------------------------------------

        v       =  2.0*ktap.astype(np.float)/1.2     # degrees of freedom
        self.se = v

        cspec = np.zeros(nf,dtype=complex)
        cohe  = np.zeros(nf,dtype=float)
        phase = np.zeros(nf,dtype=float)
        gain  = np.zeros(nf,dtype=float)
        conf  = np.zeros(nf,dtype=float)
        trf   = np.zeros(nf,dtype=complex)
        cohy  = np.zeros(nf,dtype=complex)

        for j in range(nf):

           cspec[j] = complex(sxy[j,2],sxy[j,3]) 
           cohe[j]  = (sxy[j,2]**2 + sxy[j,3]**2)/(sxy[j,0]*sxy[j,1]) 
           phase[j] = np.arctan2( sxy[j,3],  sxy[j,2]) 
           gain[j]  = np.sqrt(cohe[j]*sxy[j,1]/sxy[j,0])
           conf[j]  = 1. - ( (1.0-p)**(1./(v[j]/2. -1.0)) )
           trf[j]   = cspec[j]/sxy[j,1]
           cohy[j]  = cspec[j]/(np.sqrt(sxy[j,1]*sxy[j,0]))

        self.xspec = cspec
        self.cohe  = cohe
        self.phase = phase
        self.gain  = gain
        self.conf  = conf
        self.trf   = trf
        self.cohy  = cohy
        self.sxy   = sxy
      
        del cspec, cohe, phase, gain, conf, trf, cohy 
    #-------------------------------------------------------------------------
    # Deconvolution
    # TF = Sx/Sy 
    #    although actually we compute Sx*conj(Sy)/(Sy^2)
    # Take the IFFT to convert to the time domain. 
    # Assumes a real deconvolved signal (real input signals). 
    #-------------------------------------------------------------------------

    def mt_deconv(self): 

        """
        Compute the transfer function between two signals (cross spetrum
        already pre-computed
        """

        nf    = self.nf
        nfft  = self.nfft
        npts  = self.npts
        trf   = self.trf

        dfun  = np.zeros((npts),dtype=complex)
        dfun[0] = trf[0]
        for i in range(1,nf):
            dfun[i]      = trf[i]
            dfun[npts-i] = trf[i]
 
        dfun  = scipy.fft.ifft(dfun,npts+1) 
        dfun  = np.real(scipy.fft.ifftshift(dfun))
        dfun  = dfun[:,np.newaxis]
        dfun  = dfun/float(npts+1) 

        return dfun 

    #----------------------------------------------------------------
    # The three correlation-based estimates in the time domain
    #    correlation (cross-spectrum)
    #    deconvolution (transfer function)
    #    norm correlation (coherency)
    # Correlation:
    #    Sxy = Sx*conj(Sy)
    # Deconvolution:
    #    Sxy/Sy = Sx*conj(Sy)/Sy^2
    # Coherency
    #    Sxy/sqrt(Sx*Sy)
    #
    #----------------------------------------------------------------

    def mt_corr(self): 

        """
        Compute the time domain transfer function between two signals (cross spetrum
        already pre-computed
        """

        nf    = self.nf
        nfft  = self.nfft
        npts  = self.npts
        nfft  = self.nfft

        trf   = self.trf
        cohy  = self.cohy
        xc    = self.xspec

        dfun  = np.zeros((npts),dtype=complex)
        dfun[0] = xc[0]
        for i in range(1,nf):
            dfun[i]      = xc[i]
            dfun[npts-i] = xc[i]
        xcorr  = scipy.fft.ifft(dfun,npts+1) 
        xcorr  = np.real(scipy.fft.ifftshift(xcorr))
        xcorr  = xcorr[:,np.newaxis]
        xcorr  = xcorr/float(npts+1) 

        dfun  = np.zeros((npts),dtype=complex)
        dfun[0] = cohy[0]
        for i in range(1,nf):
            dfun[i]      = cohy[i]
            dfun[npts-i] = cohy[i]
        dcohy  = scipy.fft.ifft(dfun,npts+1) 
        dcohy  = np.real(scipy.fft.ifftshift(dcohy))
        dcohy  = dcohy[:,np.newaxis]
        dcohy  = dcohy/float(npts+1) 

        dfun  = np.zeros((npts),dtype=complex)
        dfun[0] = trf[0]
        for i in range(1,nf):
            dfun[i]      = trf[i]
            dfun[npts-i] = trf[i]
        dconv  = scipy.fft.ifft(dfun,npts+1) 
        dconv  = np.real(scipy.fft.ifftshift(dconv))
        dconv  = dconv[:,np.newaxis]
        dconv  = dconv/float(npts+1) 

        return xcorr, dcohy, dconv
 

