# Copyright 2022 Germán A. Prieto, MIT license
"""
Module with routines for univariate multitaper spectrum estimation (1D).
Contains the main MTSpec and MTSine classes where the estimates 
are made and stored.

See module mtcross for bi-variate problems

**Classes**

   * MTSpec - A class to represent Thomson's multitaper estimates
   * MTSine - A class to represent Sine Multitaper estimates

**Functions**

   * spectrogram - Computes a spectrogram with consecutive multitaper estimates.

|

"""

#-----------------------------------------------------
# Import main libraries and modules
#-----------------------------------------------------

import numpy as np
import scipy
import scipy.signal as signal
import scipy.linalg as linalg
import multitaper.utils as utils 

#-------------------------------------------------------------------------
# MTSPEC main code 
#-------------------------------------------------------------------------

class MTSpec:

    """

    .. class:: MTSpec

       A class for univariate Thomson multitaper estimates

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
          ireal  : int
             0 - real time series
             1 - complex time series

       *DPSS tapers and eigenvalues*
          vn     : ndarray [npts,kspec]
             Slepian sequences
          lamb   : ndarray [kspec]
             Eigenvalues of Slepian sequences

       *Spectral estimates*
          yk     : complex ndarray [nfft,kspec]
             eigencoefficients, fft of tapered series
          sk     : ndarray [nfft,kspec]
             eigenspectra, power spectra for each yk
          spec   : ndarray [nfft,1] 
             multitaper estimate
          se     : ndarray [nfft,1] 
             degrees of freedom of estimate
          wt     : ndarray [nfft,kspec]
             weights for each eigencoefficient at each frequency

    **Methods**

       - init     : Constructor of the MTSpec class
       - rspec    : returns the positive frequency of the spectra only
       - reshape  : reshape yk's based on F-test of line components
       - jackspec : estimate 95% confidence interval of multitaper estimate
       - qiinv    : the quadratic inverse spectral estimate
       - ftest    : F-test of line components in the spectra
       - df_spec  : dual-frequency autospectra 


    **References**
    
    Based on David J. Thomson's codes, Alan Chave and Thomson's Codes and
    partial codes from EISPACK, Robert L. Parker and Glenn Ierley from
    Scripps Institution of Oceanography. And my own Fortran90 library.

    **Notes**

    The class is in charge of estimating the adaptive weigthed 
    multitaper spectrum, as in Thomson 1982. 
    This is done by estimating the dpss (discrete prolate spheroidal 
    sequences), multiplying each of the kspec tapers with the data 
    series, take the fft, and using the adaptive scheme for a better
    estimation. 
   
    As a by product of the spectrum (spec), all intermediate steps 
    are retained, which can be used for bi-variate analysis, deconvolotuion, 
    returning to the time domain, etc. 
    By-products include the complex information in yk, the eigenspectra sk, 
    the jackknife 95% confidence intervals (spec_ci), the degrees 
    of freedom (se) and the weigths wt(nf,kspec) used.
    See below for a complete list. 
    

    **Modified**

	January 2022 (German Prieto)

   |

   """

    def __init__(self,x,nw=4,kspec=0,dt=1.0,nfft=0,iadapt=0,vn=None,lamb=None):
        """
        The constructor of the **MTSpec** class.
        
        It performs main steps in multitaper estimation, saving the
        MTSpec class variable with attributes described above.
        
        To use for first time given a time series `x`:
    
            psd = MTSpec(x,nw,kspec,dt,iadapt)
        
        *Parameters*
        
           x : ndarray [npts,]
              Time series to analyze
           nw : float, optional
              time bandwidth product, default = 4
           kspec : int, optional
              number of tapers, default = 2*nw-1
           dt : float, optional
              sampling interval of x, default = 1.0
           nfft : int, optional
              number of frequency points for FFT, allowing for padding
              default = 2*npts+1
           iadapt : int, optional
              defines methos to use, default = 0
              0 - adaptive multitaper
              1 - unweighted, wt =1 for all tapers
              2 - wt by the eigenvalue of DPSS
           vn : ndarray [npts,kspec], optional
              Slepian sequences, can be precomputed to save time
           lamb : ndarray [kspec], optional
              Eigenvalues of DPSS, can be precomputed to save time

        |

        """

        #-----------------------------------------------------
        # Check number of tapers
        #-----------------------------------------------------

        if (kspec < 1):
            kspec = np.int(np.round(2*nw-1))

        #-----------------------------------------------------
        # Check dimensions of input vector
        #-----------------------------------------------------

        ndim  = x.ndim
        if (ndim>2):
            raise ValueError("Array cannot by 3D")
        elif (ndim==1):
            x = x[:, np.newaxis]
            ndim = x.ndim 
        ncol = x.shape[1]
        npts = x.shape[0]

        if (ncol>1):
            raise ValueError("Array must be a single column")

        #-----------------------------------------------------
        # Check if real or complex
        #-----------------------------------------------------

        ireal = 0
        treal = np.iscomplex(x);
        if (treal.any()):
            ireal = 1

        #-----------------------------------------------------------------
        # Detrend, get variance
        #-----------------------------------------------------------------

        x     = signal.detrend(x,axis=0,type='constant')
        xvar  = np.var(x)

        #-----------------------------------------------------------------
        # Define other parameters (nfft, nf, freq vector)
        #-----------------------------------------------------------------
       
        nfft = np.int(nfft)
        if (nfft < npts):
            nfft = 2*npts + 1
        if (nfft%2 == 0):
            nf = int(nfft/2 + 1)
        else:
            nf = int((nfft+1)/2)

        freq       = scipy.fft.fftfreq(nfft,dt)
        freq[nf-1] = abs(freq[nf-1])    # python has fnyq negative
        df         = freq[2]-freq[1]
        freq       = freq[:, np.newaxis]

        #------------------------------------------------------
        # Put main parameters in self
        #------------------------------------------------------

        self.x      = x
        self.xvar   = xvar
        self.freq   = freq
        self.dt     = dt
        self.df     = df
        self.nf     = nf
        self.nw     = nw
        self.kspec  = kspec
        self.nfft   = nfft
        self.npts   = npts
        self.iadapt = iadapt
        self.ireal  = ireal

        #-----------------------------------------------------
        # Calculate DPSS (tapers)
        #-----------------------------------------------------

        if (vn is None):
            vn, lamb = utils.dpss(npts,nw,kspec)
            self.vn = vn
            self.lamb = lamb
            del vn, lamb 
        else:
            npts2  = np.shape(vn)[0]
            kspec2 = np.shape(vn)[1]
            if (npts2 != npts or kspec2 != kspec):
                vn, lamb = utils.dpss(npts,nw,kspec)
                self.vn = vn
                self.lamb = lamb
                del vn, lamb
            else:
                self.vn = vn
                self.lamb = lamb
                del vn, lamb
                
        #-----------------------------------------------------------------
        # Get eigenspectra
        #-----------------------------------------------------------------

        yk, sk  = utils.eigenspec(self.x,self.vn,self.lamb,self.nfft)
        self.yk = yk
        self.sk = sk
        del sk, yk

        #-----------------------------------------------------------------
        # Calculate adaptive spectrum
        #-----------------------------------------------------------------

        spec, se, wt = utils.adaptspec(self.yk,self.sk,self.lamb,self.iadapt)
        sscal = np.sum(spec)*df
        sscal = xvar/sscal
        spec  = sscal*spec

        self.spec = spec
        self.se   = se
        self.wt   = wt
        del se, wt, spec

    #-------------------------------------------------------------------------
    # Finished INIT mvspec
    #-------------------------------------------------------------------------

    #----------------------------------------------------------------
    # Return positive freq only (if real)
    #----------------------------------------------------------------

    def rspec(self,*args):
        """
        Returns the spetra at positive frequencies, checking that 
        a real input signal was used.

        *Parameters*
        
           args : ndarray
              another array to return the positive frequencies. 
              Could be qispec, spec_ci, etc.

        |

        """

        nargs = len(args)
        if (self.ireal==1):
           
            print("Input signal is complex, returns entire spectrum") 

            if (nargs>0):
                return self.freq,args

            return self.freq, self.spec

        elif (self.ireal==0):
            nf       = self.nf
            freq     = np.zeros((nf,1), dtype=float)
            freq     = self.freq[0:nf]
           
            # Check args, and create new tuple with correct size
            if (nargs>0):
                argout = tuple()
                tup1   = (freq,)
                argout = argout + tup1
                for i in range(nargs):
                    t_in  = args[i]
                    ncol  = np.shape(t_in)[1]
                    t_out = np.zeros((nf,ncol), dtype=float)
                    t_out[0:nf,:] = 2.0*t_in[0:nf,:]
                    t_out[0,:]    = 0.5*t_out[0,:]
                    if (self.nfft%2==0):
                        t_out[nf-1,:]   = 0.5*t_out[nf-1,:]
                    tup1  = (t_out,)
                    argout = argout + tup1
                return argout

            # Spectrum, Double power, except at 0, fnyq
            spec     = np.zeros((nf,1), dtype=float)
            spec[0:nf,0]   = 2.0*self.spec[0:nf,0]
            spec[0,0]      = 0.5*spec[0,0]
            if (self.nfft%2==0):
                spec[nf-1,0]   = 0.5*spec[nf-1,0]

            return freq,spec

           

    #----------------------------------------------------------------
    # Remove lines, save spectrum without lines
    #----------------------------------------------------------------

    def reshape(self,fcrit=0.95,p=None):   
        """
        Reshape eigenft's (yk) around significant spectral lines
        The "significant" means above fcritical probability (0.95)
        If probability is large at neighbouring frequencies, I will 
        only remove the largest probability energy. 

        Returns recalculated yk, sk, spec, wt, and se

        *Parameters*
        
           fcrit : float optional
              Probability value over which to reshape, default = 0.95
           p : ndarray optional [nfft] 
              F-test probabilities to find fcritical
              If None, it will be calculated

        *Returns*
        
           respec : ndarray [nfft]
              The reshaped PSD estimate 
           spec : ndarray [nfft]
              the PSD without the line components
           yk : ndarray [nfft,kspec]
              the eigenft's without line components
           sline : ndarray [nfft]
              the PSD of the line components only

        *Calls*
        
           utils.yk_reshape

        |

        """

        if (p is None):
            yk,sline = utils.yk_reshape(self.yk,self.vn,fcrit=fcrit)
        else:
            yk,sline = utils.yk_reshape(self.yk,self.vn,p=p,fcrit=fcrit)


        #-----------------------------------------------------------------
        # Calculate adaptive spectrum
        #-----------------------------------------------------------------

        sk           = np.abs(yk)**2
        spec, se, wt = utils.adaptspec(yk,sk,self.lamb,self.iadapt)

        #-----------------------------------------------------------------
        # For reshaped, add line components
        #-----------------------------------------------------------------

        respec = spec + sline

        #-----------------------------------------------------------------
        # Normalize energy, Parseval's with lines. 
        #-----------------------------------------------------------------

        sscal   = np.sum(respec)*self.df
        sscal   = self.xvar/sscal
        respec  = sscal*respec
        spec    = sscal*spec

        return respec, spec, yk, sline

    #-------------------------------------------------------------------------
    # jackspec
    #-------------------------------------------------------------------------

    def jackspec(self):

        """
        code to calculate adaptively weighted jackknifed 95% confidence limits

        *Returns*
        
           spec_ci : ndarray [nfft,2]
              real array of jackknife error estimates, with 5 and 95%
              confidence intervals of the spectrum.

        *Calls*
        
           utils.jackspec

        """

        spec_ci = utils.jackspec(self.spec,self.sk,self.wt,self.se)

        return spec_ci 

    #-------------------------------------------------------------------------
    # end jackspec
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # qiinv
    #-------------------------------------------------------------------------
    
    def qiinv(self): 
        """
        Function to calculate the Quadratic Spectrum using the method 
        developed by Prieto et al. (2007).   
        
        The first 2 derivatives of the spectrum are estimated and the 
        bias associated with curvature (2nd derivative) is reduced. 

        Calculate the Stationary Inverse Theory Spectrum.
        Basically, compute the spectrum inside the innerband. 
  
        This approach is very similar to D.J. Thomson (1990).

        *Returns*
        
           qispec : ndarray [nfft,0]
              the QI spectrum estimate
           ds : ndarray [nfft,0]	
              the estimate of the first derivative
           dds : ndarray [nfft,0]	
              the estimate of the second derivative

        *References*
        
           G. A. Prieto, R. L. Parker, D. J. Thomson, F. L. Vernon, 
           and R. L. Graham (2007), Reducing the bias of multitaper 
           spectrum estimates,  Geophys. J. Int., 171, 1269-1281. 
           doi: 10.1111/j.1365-246X.2007.03592.x.
  
        *Calls*
        
           utils.qiinv
        
        | 

        """
   
        qispec, ds, dds = utils.qiinv(self.spec,self.yk,
                                      self.wt,self.vn,self.lamb,
                                      self.nw)

        #----------------------------------------------------------------------
        # Normalize spectrum and derivatives
        #----------------------------------------------------------------------
 
        qisscal = np.sum(qispec)*self.df
        qisscal = self.xvar/qisscal

        qispec = qispec*qisscal
        ds     = ds*qisscal
        dds    = dds*qisscal
 
        return qispec, ds, dds
    
    #-------------------------------------------------------------------------
    # end qiinv
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # ftest
    #-------------------------------------------------------------------------
    
    def ftest(self):
        """
        Performs the F test for a line component

        Computes F-test for single spectral line components
        at the frequency bins given in the MTSpec class.

        **Returns**
        
  	    F : ndarray [nfft]
            vector of f test values, real
  	    p : ndarray [nfft]
            vector with probability of line component

        **Calls**
        
        utils.f_test
        
        |
        
        """
   
        F,p   = utils.ftest(self.vn, self.yk)
 
        return F, p
    
    #-------------------------------------------------------------------------
    # end ftest 
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # df spectrum
    #-------------------------------------------------------------------------
    
    def df_spec(self):
        """
        Performs the dual-frequency spectrum of a signal with itself.

        *Returns*
        
           df_spec : ndarray complex, 2D (nf,nf)
              the complex dual-frequency cross-spectrum. Not normalized
           df_cohe : ndarray, 2D (nf,nf)
              MSC, dual-freq coherence matrix. Normalized (0.0,1.0)
           df_phase : ndarray, 2D (nf,nf)
              the dual-frequency phase

        *Calls*
        
           utils.df_spec

        |

        """
   
        df_spec, df_cohe, df_phase = utils.df_spec(self)
 
        return df_spec, df_cohe, df_phase
    
    #-------------------------------------------------------------------------
    # end df spetrum
    #-------------------------------------------------------------------------

#------------------------------------------------------------------------------    
# End CLASS MTSPEC
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# CLASS MTSINE
#------------------------------------------------------------------------------

class MTSine:

    """
    .. class:: MTSpec

       A class for univariate Thomson multitaper estimates

    **Attibutes**

    *Parameters*
       npts   : int
          number of points of time series
       nfft   : int
          number of points of FFT. nfft = 2*npts

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
       spec   : ndarray [nfft,1] 
          multitaper estimate
       err     : ndarray [nfft,2]
          1-std confidence interval of spectral estimate
          simple dof estimate

    **Notes**
    
    The class is in charge of estimating the adaptive sine 
    multitaper as in Riedel and Sidorenko (1995). 
    This is done by performing a MSE adaptive estimation. First
    a pilot spectral estimate is used, and S" is estimated, in 
    order to get te number of tapers to use, using (13) of 
    R & S for a min square error spectrum.

    |

    """

    def __init__(self,x,ntap=0,ntimes=0,fact=1.0,dt=1.0):
        """
        Performs the PSD estimation by the sine multitaper method
     
        **Parameters**
        
           x : ndarray [npts]	
              real, data vector
      	   ntap : int, optional
              constant number of tapers (def = 0)
      	   ntimes : int, optional
              number of iterations to perform
      	   fact : float, optional
              degree of smoothing (def = 1.)
       	   dt : float
              sampling interval of time series
      
        
        **Notes**
        
        This function is in charge of estimating the adaptive sine 
        multitaper as in Riedel and Sidorenko (1995). 
        This is done by performing a MSE adaptive estimation. First
        a pilot spectral estimate is used, and S" is estimated, in 
        order to get te number of tapers to use, using (13) of 
        R & S for a min square error spectrum. 

        Unlike the prolate spheroidal multitapers, the sine multitaper 
        adaptive process introduces a variable resolution and error in 
        the frequency domain. Complete error information is contained 
        in the output variables file as the corridor of 1-standard-deviation 
        errors, and in K, the number of tapers used at each frequency.
        The errors are estimated in the simplest way, from the number of 
        degrees of freedom (two per taper), not by jack-knifing. The 
        frequency resolution is found from K*fN/Nf where fN is the Nyquist 
        frequency and Nf is the number of frequencies estimated.
        The adaptive process used is as follows. A quadratic fit to the
        log PSD within an adaptively determined frequency band is used 
        to find an estimate of the local second derivative of the 
        spectrum. This is used in an equation like R & S (13) for the 
        MSE taper number, with the difference that a parabolic weighting 
        is applied with increasing taper order. Because the FFTs of the 
        tapered series can be found by resampling the FFT of the original 
        time series (doubled in length and padded with zeros) only one FFT 
        is required per series, no matter how many tapers are used. This 
        makes the program fast. Compared with the Thomson multitaper 
        programs, this code is not only fast but simple and short. The 
        spectra associated with the sine tapers are weighted before 
        averaging with a parabolically varying weight. The expression 
        for the optimal number of tapers given by R & S must be modified
        since it gives an unbounded result near points where S" vanishes,
        which happens at many points in most spectra. This program 
        restricts the rate of growth of the number of tapers so that a 
        neighboring covering interval estimate is never completely 
        contained in the next such interval.

        This method SHOULD not be used for sharp cutoffs or deep 
        valleys, or small sample sizes. Instead use Thomson multitaper
        in mtspec in this same library. 

        **References**
        
        Riedel and Sidorenko, IEEE Tr. Sig. Pr, 43, 188, 1995

        Based on Bob Parker psd.f codes. Most of the comments come 
        his documentation as well.

      	
        **Modified**
    
      	September 22 2005


        **Calls**
        
        utils.quick, utils.adapt
     
        | 
        """

        #-----------------------------------------------------
        # Check dimensions of input vector
        #-----------------------------------------------------

        ndim  = x.ndim
        if (ndim>2):
            raise ValueError("Array cannot by 3D")
        elif (ndim==1):
            x = x[:, np.newaxis]
            ndim = x.ndim 
        ncol = x.shape[1]
        npts = x.shape[0]

        if (ncol>1):
            raise ValueError("Array must be a single column")


        #-----------------------------------------------------
        # Check if real or complex
        #-----------------------------------------------------

        ireal = 0
        treal = np.iscomplex(x);
        if (treal.any()):
            ireal = 1

        #-----------------------------------------------------------------
        # Detrend, get variance
        #-----------------------------------------------------------------

        x     = signal.detrend(x,axis=0,type='constant')
        xvar  = np.var(x)

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
        # Setdefaults
        #------------------------------------------------------

        if (ntap < 2): 
            ntap = 0
        if (ntimes <= 0):
            ntimes = 2 
        if (fact <= 0.):
            fact = 1.

        #------------------------------------------------------
        # Put main parameters in self
        #------------------------------------------------------

        self.x      = x
        self.xvar   = xvar
        self.freq   = freq
        self.dt     = dt
        self.df     = df
        self.nf     = nf
        self.ntap   = ntap
        self.nfft   = nfft
        self.npts   = npts
        self.ntimes = ntimes
        self.ireal  = ireal

        #-----------------------------------------------------
        # Get the FFT of the time series
        #    Only one FFTY is required
        #-----------------------------------------------------

        fx  = scipy.fft.fft(x,axis=0,n=nfft)

        #-----------------------------------------------------
        #  Check if constant tapers or adaptive method
        #-----------------------------------------------------

        if (ntap>0):

            #  Estimate uniform taper PSD
            spec, kopt = utils.squick(nfft,fx,nf,ntap)

            sscal = np.sum(spec)*df
            sscal = xvar/sscal
            spec  = sscal*spec

            self.spec = spec
            self.kopt = kopt

        else:

            itap = int(3.0 + np.sqrt(fact*float(npts))/5.0)
            spec, kopt = utils.sadapt(nfft,fx,nf,df,itap,
                               ntimes,fact)

            sscal = np.sum(spec)*df
            sscal = xvar/sscal
            spec  = sscal*spec

            self.spec = spec
            self.kopt = kopt

        # end if ntap>0 

        #----------------------------------------------------------------
        #  Error estimate
        #  The 5 - 95% confidence limits are estimated using the 
        #  approximation of Chambers et al, 1983 Graphical Methods
        #  for data Analysis. See also Percival and Walden p. 256, 1993
        #  The 1.2 factor comes from the parabolic weighting.
        #----------------------------------------------------------------

        err = np.zeros((nf,2),dtype=float)

        std = spec/ np.sqrt(kopt/1.2)   # 1 standard deviation
        v   = 2.0*kopt/1.2              # degrees of freedom

        err1 =  spec / (1-2./(9.0*v)-1.96*np.sqrt(2./(9.0*v)))**3
        err2 =  spec / (1-2./(9.0*v)+1.96*np.sqrt(2./(9.0*v)))**3;
        err[:,0] = err1
        err[:,1] = err2

        self.err = err

    #-------------------------------------------------------------------------
    # Finished INIT mtsine
    #-------------------------------------------------------------------------

 
#------------------------------------------------------------------------------
# end CLASS MTSINE
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

#--------------------------
# Spectrogram
#--------------------------

def spectrogram(data,dt,twin,olap=0.5,nw=3.5,kspec=5,fmin=0.0,fmax=-1.0,iadapt=0):
    """
    Computes a spectrogram with consecutive multitaper estimates.
    Returns both Thomson's multitaper and the Quadratic multitaper estimate

    **Parameters**
    
    data : array_like (npts,)
        Time series or sequence
    dt : float
        Sampling interval in seconds of the time series.
    twin : float
        Time duration in seconds of each segment for a single multitaper estimate.
    olap : float, optional
        Overlap requested for the segment in building the spectrogram. 
        Defaults = 0.5, values must be (0.0 - 0.99).
        Overlap rounds to the nearest integer point. 
    nw : float, optional
        Time-bandwidth product for Thomson's multitaper algorithm.
        Default = 3.5
    kspec : int, optional
        Number of tapers for avearaging the multitaper estimate.
        Default = 5
    fmin : float, optional
        Minimum frequency to estimate the spectrogram, otherwise returns the 
        entire spectrogram matrix.
        Default = 0.0 Hz
    fmax : float, optional
        Maximum frequency to estimate the spectrogram, otherwise returns the
        entire spectrogram matrix. 
        Default = 0.5/dt Hz (Nyquist frequency)
    iadapt : integer, optional
        User defined, determines which method for multitaper averaging to use. 
        Default = 0
        0 - Adaptive multitaper
        1 - Eigenvalue weights
        2 - Constant weighting

    **Returns**
    
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Quad : ndarray
        Spectrogram of x using the quadratic multitaper estimate.
    MT : ndarray
        Spectrogram of x using Thomson's multitaper estimate.
    
    By default, the last axis of Quad/MT corresponds to the segment times.

    **See Also**
    
    MTSpec: Multitaper estimate of a time series.

    **Notes**
    
    The code assumes a real input signals and thus mainly returns the positive 
    frequencies. For a complex input signals, code qould require adaptation.  

    **References**
    
       Prieto, G.A. (2022). The multitaper spectrum analysis package in Python.
       Seism. Res. Lett In review. 

    **Examples**
   
    To do

    |

    """
    
    if (fmax<=0.0):
        fmax = 0.5/dt
    
    nwin  = int(np.round(twin/dt))
    if (olap<=0.0):
        njump = nwin
    else:
        njump = int(np.round(twin*(1.0-olap))/dt)

    npts  = np.size(data)
    nmax  = npts-nwin
    nvec  = np.arange(0,nmax,njump)
    t     = nvec*dt
    nspec = len(nvec)

    print('Window length %5.1fs and overlap %2.0f%%' %(twin, olap*100))
    print('Total number of spectral estimates', nspec)
    print('Frequency band of interest (%5.2f-%5.2f)Hz' %(fmin, fmax))

    vn,theta = utils.dpss(nwin,nw,kspec)
    for i in range(nspec):
        if ((i+1)%10==0):
            print('Loop ',i+1,' of ',nspec)

        i1  = nvec[i]
        i2  = i1+nwin
        x   = data[i1:i2+1]

        psd = MTSpec(x,nw,kspec,dt,iadapt=iadapt,
                            vn=vn,lamb=theta)

        freq2   = psd.freq
        spec    = psd.spec 
        qispec  = psd.qiinv()[0]   

        nf         = len(freq2)

        if (i==0):
            fres   = np.where((freq2>=fmin) & (freq2<=fmax))[0]
            nf     = len(fres)
            f      = freq2[fres]
            Quad   = np.zeros((nf,nspec),dtype=float)
            MT     = np.zeros((nf,nspec),dtype=float)
            print('Total frequency points %i' %(nf))

        Quad[:,i]  = qispec[fres,0]
        MT[:,i]    = spec[fres,0] 
    
    return t,f,Quad,MT
