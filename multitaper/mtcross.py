# Copyright 2022 Germán A. Prieto, MIT license
"""
Module that contains all multivariate-multitaper codes. 

Contains:
   - mt_cohe
   - mt_deconv
   - To do: wv_spec

Module with routines for bi-variate multitaper spectrum estimation.
Contains the main MTCross and SineCross classes where the estimates 
are made and stored. 

It takes univariate classes MTSpec and MTSine for estimating coherence
tranfer functions, etc. 

See module mtspec for univariate problems

**Classes**

    * MTCross   - A class to represent Thomson's multitaper cross-spectra
    * SineCross - A class to represent Sine Multitaper cross-spectra

**Functions**
    None

|

"""

#-----------------------------------------------------
# Import main libraries and modules
#-----------------------------------------------------

import numpy as np
import scipy
from scipy import signal
import scipy.linalg      as linalg
import multitaper.utils      as utils 
import multitaper.mtspec     as spec

class MTCross:

    """
    .. class:: MTCross

        A class for bi-variate Thomson multitaper estimates

    **Attibutes**

    *Parameters*

    npts   : int
        number of points of time series
    nfft   : int
        number of points of FFT. Dafault adds padding. 
    nw     : flaot
        time-bandwidth product
    kspec  : int
        number of tapers to use

    *Time series*

    x      : ndarray [npts]
        time series
    xvar   : float
        variance of time series
    dt     : float
        sampling interval

    *Frequency vector*

    nf     : int
        number of unique frequency points of spectral 
        estimate, assuming real time series
    freq   : ndarray [nfft]
        frequency vector in Hz
    df     : float
        frequncy sampling interval

    *Method*

    iadapt : int
        defines methos to use
        0 - adaptive multitaper
        1 - unweighted, wt =1 for all tapers
        2 - wt by the eigenvalue of DPSS

    *Spectral estimates*

    Sxx : ndarray [nfft]
        Power spectrum of x time series
    Syy : ndarray [nfft]
        Power spectrum of y time series
    Sxy : ndarray, complex [nfft]
        Coss-spectrum of x, y series
    cohe  : ndarray [nfft]
        MSC, freq coherence. Normalized (0.0,1.0)
    phase : ndarray [nfft]
        the phase of the cross-spectrum    
    cohy : ndarray, complex [nfft]
        the complex coherency, normalized cross-spectrum 
    trf  : ndarray, compolex [nfft]
        the transfer function Sxy/(Syy_wl), with water-level optional
    se : ndarray [nfft,1] 
        degrees of freedom of estimate
    wt : ndarray [nfft,kspec]
        weights for each eigencoefficient at each frequency

    **Methods**

       * init      : Constructor of the MTCross class
       * mt_deconv : Perform the deconvolution from the self.trf, by iFFT
       * mt_corr   : compute time-domain via iFFT of cross-spectrum, 
                     coherency, and transfer function

    **Modified**
    
	German Prieto
	January 2022

    |

    """

    def __init__(self,x,y,nw=4,kspec=0,dt=1.0,nfft=0,iadapt=0,wl=0.0):
        """
        The constructor of the MTCross class.

        It performs main steps in bi-variate multitaper estimation, 
        including cross-spectrum, coherency and transfer function.
        
        MTCross class variable with attributes described above. 

        **Parameters**
        
        x : MTSpec class, or ndarray [npts,]
            Time series signal x.
            If ndarray, the MTSpec class is created.
        y : MTSpec class, or ndarray [npts,]
            Time series signal x
            If ndarray, the MTSpec class is created.
        nw : float, optional
            time bandwidth product, default = 4
            Only needed if x,y are ndarray
        kspec : int, optional
            number of tapers, default = 2*nw-1
            Only needed if x,y are ndarray
        dt : float, optional
            sampling interval of x, default = 1.0
            Only needed if x,y are ndarray
        nfft : int, optional
            number of frequency points for FFT, allowing for padding
            default = 2*npts+1
            Only needed if x,y are ndarray
        iadapt : int, optional
            defines methos to use, default = 0
            0 - adaptive multitaper
            1 - unweighted, wt =1 for all tapers
            2 - wt by the eigenvalue of DPSS
        wl : float, optional
            water-level for stabilizing deconvolution (transfer function).
            defined as proportion of mean power of Syy

        |

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

            x = spec.MTSpec(x,nw,kspec,dt,nfft,iadapt=iadapt)
            y = spec.MTSpec(y,nw,kspec,dt,nfft,iadapt=iadapt,vn=x.vn,lamb=x.lamb)

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
        self.wt     = wt

        del Sxx, Syy, Sxy, cohe, phase, se, wt

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
        Generate a deconvolution between two time series, returning
        the time-domain signal.
        
        MTCross has already pre-computed the cross-spectrum and 
        the transfer function. 

        **Returns**
        
        dfun : ndarray [nfft]
            time domain of the transfer function. 
            delay time t=0 in centered in the middle.

        **References**
        
        The code more or less follows the paper
        Receiver Functions from multiple-taper spectral corre-
        lation estimates. J. Park and V. Levin., BSSA 90#6 1507-1520

        It also uses the code based on dual frequency I created in
        GA Prieto, Vernon, FL , Masters, G, and Thomson, DJ (2005), 
        Multitaper Wigner-Ville Spectrum for Detecting Dispersive 
        Signals from Earthquake Records, Proceedings of the 
        Thirty-Ninth Asilomar Conference on Signals, Systems, and 
        Computers, Pacific Grove, CA., pp 938-941. 

        | 

        """

        nfft  = self.nfft
        trf   = self.trf

        dfun  = scipy.fft.ifft(trf[:,0],nfft) 
        dfun  = np.real(scipy.fft.fftshift(dfun))
        dfun  = dfun[:,np.nexaxis]
        dfun  = dfun/float(nfft) 

        return dfun 


    def mt_corr(self): 

        """
        Compute time-domain via iFFT of cross-spectrum, 
        coherency, and transfer function
 
        Cross spectrum, coherency and transfer function 
        already pre-computed in MTCross.

        **Returns**
        
        xcorr : ndarray [nfft]
            time domain of the transfer function. 
        dcohy : ndarray [nfft]
            time domain of the transfer function. 
        dfun : ndarray [nfft]
            time domain of the transfer function. 
            
        Delay time t=0 in centered in the middle.

        **Notes**
        
        The three correlation-based estimates in the time domain
            - correlation (cross-spectrum)
            - deconvolution (transfer function)
            - norm correlation (coherency)
        Correlation:
            - Sxy = Sx*conj(Sy)
        Deconvolution:
            - Sxy/Sy = Sx*conj(Sy)/Sy^2
        Coherency
            - Sxy/sqrt(Sx*Sy)
        
        | 

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

class SineCross:

    """
    .. class:: SineCross

        A class for bi-variate Sine multitaper estimates

    **Attibutes**
    

    *Parameters*
       npts   : int
          number of points of time series
       nfft   : int
          number of points of FFT. nfft = 2*npts

    *Time series*
       x      : ndarray [npts]
          time series x
       xvar   : float
          variance of x time series
       y      : ndarray [npts]
          time series y
       yvar   : float
          variance of y time series
       dt     : float
          sampling interval

    *Frequency vector*
       nf     : int
          number of unique frequency points of spectral 
          estimate, assuming real time series
       freq   : ndarray [nfft]
          frequency vector in Hz
       df     : float
          frequncy sampling interval

    *Method*
       ntap   : int
          fixed number of tapers
          if ntap<0, use kopt
       kopt   : ndarray [nfft,1] 
          number of tapers at each frequency
       ntimes : int
          number of max iterations to perform
       ireal  : int
          0 - real time series
          1 - complex time series

    *Spectral estimates*
       cspec : ndarray, complex [nfft]
          Coss-spectrum of x, y series
       sxy : ndarray, complex [nfft]
          Coss-spectrum of x, y series
       cohe  : ndarray [nfft]
          MSC, freq coherence. Normalized (0.0,1.0)
       phase : ndarray [nfft]
          the phase of the cross-spectrum    
       gain : ndarray [nfft]
          the gain for the two spectra    
       cohy : ndarray, complex [nfft]
          the complex coherency, normalized cross-spectrum 
       trf  : ndarray, compolex [nfft]
          the transfer function Sxy/(Syy_wl), with water-level optional
       se : ndarray [nfft,1] 
          degrees of freedom of estimate
       conf : ndarray [nfft,]
          confidence in cross-spectrum at each frequency

    **Methods**

    - init      : Constructor of the SineCross class
    - mt_deconv : Perform the deconvolution from the self.trf, by iFFT
    - mt_corr   : compute time-domain via iFFT of cross-spectrum, 
                  coherency, and transfer function

    **Modified**

	January 2022, German A. Prieto

    |

    """


    def __init__(self,x,y,ntap=0,ntimes=0,fact=1.0,dt=1.0,p=0.95):

        """
        Performs the coherence and cross-spectrum estimation 
        by the sine multitaper method.


        **Parameters**

        x : MTSine class, or ndarray [npts,]
            Time series signal x.
            If ndarray, the MTSpec class is created.
        y : MTSine class, or ndarray [npts,]
            Time series signal x
            If ndarray, the MTSpec class is created.
      	ntap : int, optional
            constant number of tapers (def = 0)
      	ntimes : int, optional
            number of iterations to perform
      	fact : float, optional
            degree of smoothing (def = 1.)
       	dt : float, optional
            sampling interval of time series
        p : float, optional
            proportion for confidence intervale estimation

        **References**
        
        Riedel and Sidorenko, IEEE Tr. Sig. Pr, 43, 188, 1995

        Based on Bob Parker psd.f and cross.f codes. Most of the comments 
        come from his documentation as well.

        |

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
        self.y      = y
        self.yvar   = yvar
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
        Generate a deconvolution between two time series, returning
        the time-domain signal.
        
        SineCross has already pre-computed the cross-spectrum and 
        the transfer function. 

        **Returns**
        
        dfun : ndarray [nfft]
            time domain of the transfer function. 
            delay time t=0 in centered in the middle.

        |

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

    def mt_corr(self): 

        
        """
        Compute time-domain via iFFT of cross-spectrum, 
        coherency, and transfer function
 
        Cross spectrum, coherency and transfer function 
        already pre-computed in SineCross class.

        **Returns**
        
        xcorr : ndarray [nfft]
            time domain of the transfer function. 
        dcohy : ndarray [nfft]
            time domain of the transfer function. 
        dfun : ndarray [nfft]
            time domain of the transfer function. 
            
        Delay time t=0 in centered in the middle.

        **Notes**
        
        The three correlation-based estimates in the time domain
            - correlation (cross-spectrum)
            - deconvolution (transfer function)
            - norm correlation (coherency)
        Correlation:
            - Sxy = Sx*conj(Sy)
        Deconvolution:
            - Sxy/Sy = Sx*conj(Sy)/Sy^2
        Coherency
            - Sxy/sqrt(Sx*Sy)
        
        |

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
 

