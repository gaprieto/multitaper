"""
Module with all routines for multitaper 
spectrum estimation (1D). 
Contains:
   mtspec - calculate the Thomson and/or Quadratic multitaper 
"""

#-----------------------------------------------------
# Import main libraries and modules
#-----------------------------------------------------

import numpy as np
import scipy
import scipy.signal as signal
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import multitaper.utils as utils 
import time

#-------------------------------------------------------------------------
# MTSPEC main code 
#-------------------------------------------------------------------------

class mtspec:

    def __init__(self,x,nw=4,kspec=0,dt=1.0,nfft=0,iadapt=0,vn=None,lamb=None):
        """
        put some notes here
        Only works so far, on a single dimensional direction
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
        If input signal is real, returns the positive frequencies 
        only. It keeps the power correct (twice power in positive 
        frequencies, except at freq=0 and freq=fnyq) 
        
        If input is another vector (qispec, spec_ci, etc) then 
        only that vector is processed. 
        Vector must be real (must be power spectrum or derivatives)
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

    def reshape(self,fcrit=0.95,p=None):   #p_in,yk_in,vn,fcrit=0.95):
        """
        reshape the yk's based on the F-test of line compenents
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

        spec_ci = utils.jackspec(self.spec,self.sk,self.wt,self.se)

        return spec_ci 

    #-------------------------------------------------------------------------
    # end jackspec
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # qiinv
    #-------------------------------------------------------------------------
    
    def qiinv(self): #spec,yk,wt,vn,lamb,nw):
    
        """
        Function to calculate the Quadratic Spectrum using the method 
        developed by Prieto et al. (2007).   
        The first 2 derivatives of the spectrum are estimated and the 
        bias associated with curvature (2nd derivative) is reduced. 
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
    
    def ftest(self): #vn,yk):
        """
        Performs the F test for a line component
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

class mtsine:

    def __init__(self,x,ntap=0,ntimes=0,fact=1.0,dt=1.0):
        """
        Performs the PSD estimation by the sie multitaper method of
        Riedel and Sidorenko, IEEE Tr. Sig. Pr, 43, 188, 1995
      
        Based on Bob Parker psd.f codes. Most of the comments come 
        his documentation as well.
      
        Last Modified:
      	German Prieto
      	September 22 2005
      	
        The subroutine is in charge of estimating the adaptive sine 
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
      
        INPUT
      	  x[npts]	real, data vector
      	  ntap		integer, constant number of tapers (def = 0)
      	  ntimes	integer, number of iterations to perform
      	  fact		real, degree of smoothing (def = 1.)
       	  dt		real, sampling rate of time series
      
        OUTPUT
      	  freq(nfft)	real vector with frequency bins
      	  spec(nfft)	real vector with the adaptive estimated spectrum
        Note: NFFT is 2*NPTS

        OPTIONAL OUTPUT
          kopt(nf)	integer, number of taper per freq point
      	  err(nf,2)	1-std errors (simple dof estimate)
      	
        calls quick, adapt
      
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


