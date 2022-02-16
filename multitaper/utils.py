# Copyright 2022 Germán A. Prieto, MIT license
"""
Module with all the definitions (routines) of general use 
of the multitaper routines. 

Contains:
   *  set_xint   - setup Ierly's quadrature
   *  xint       - Quadrature by Ierley's method of Chebychev sampling.
   *  dpss_ev    - Recalculate the DPSS eigenvalues using Quadrature
   *  dpss       - calculate the DPSS for given NW, NPTS
   *  eigenspec  - calculate eigenspectra using DPSS sequences. 
   *  adaptspec  - calculate adaptively weighted power spectrum
   *  jackspec   - calculate adaptively weighted jackknifed 95% confidence limits
   *  qiinv      - calculate the Stationary Inverse Theory Spectrum.
   *  ftest      - performs the F-test for a line component
   *  yk_reshape - reshape eigenft's around significant spectral lines
   *  wt2dof     - calculate the d.o.f. of the multitaper
   *  df_spec    - Dual frequency spectrum, using two MTSPEC classes to compute.  
   *  sft        - the slow Fourier transform
   *  squick     - for sine multitaper, constructs average multitaper
   *  squick2    - for sine multitaper, constructs average multitaper, 2 signals
   *  sadapt     - for sine multitaper, adaptive estimation of # of tapers
   *  sadapt2    - for sine multitaper, same but for 2 signals
   *  north      - for sine multitaper, derivatives of spectrum
   *  curb       - for sine multitaper, clips # of tapers
   *  get_data   - download data and load into numpy array 

|

"""

#-----------------------------------------------------
# Import main libraries and modules
#-----------------------------------------------------

import numpy as np
import scipy
from scipy import signal
import scipy.linalg as linalg
import scipy.interpolate as interp
import scipy.optimize as optim
import os



#-------------------------------------------------------------------------
# SET_XINT - Set up weights and sample points for Ierly quadrature
#-------------------------------------------------------------------------

def set_xint(ising):
    """
    Sets up weights and sample points for Ierley quadrature,
    
    Slightly changed from original code, to avoid using common
    blocks. Also avoided using some go to statements, not needed.
    
    *Parameters*
    
    ising : integer
        ising=1 	
            integrand is analytic in closed interval
        ising=2	
            integrand may have bounded singularities 
            at end points
       
    *Returns*
    
    w : ndarray (nomx,lomx+1)
        weights
    x : sample points (lomx+1)	
        sample points
    
    lomx=number of samples = 2**nomx

    *Modified*
    
    November 2004 (German A. Prieto)

    |

    """

    nomx = 8
    lomx = 256
    w = np.zeros((nomx,lomx+1),dtype=float)
    x = np.zeros(lomx+1,dtype=float)

    pi = np.pi
    n  = 2

    for index in range(1,nomx+1):
        n  = 2*n
        nx = n-2
        if (index == 1):
            nx=4
      
        pin   = pi/float(n)
        nhalf = int(n/2) 
        for i in range(nhalf+1):
            t  = float(i)*pin
            si = 0.0
            for k in range(0,nx+1,2):
                ck=4.0
                if (k == 0):
                    ck=2.0
                rk=float(k)
                si=si+ck*np.cos(rk*t)/(1.0-rk*rk)

            if (i==0 or i==nhalf):
                si=0.5*si

            t = np.cos(t)

            if (ising == 2): 
                t=0.5*pi*(1.0 +t)
                si=si*0.5 * np.sin(t)*pi
                t=np.cos(t)
                x[i]          = 0.5 *(1.0 +t)
                w[index-1, i] = 0.5 *si/float(n)
            elif (ising == 1):
                x[i]         = 0.5 *(1.0 +t)
                w[index-1,i] = 0.5 *si/float(n)
        # end i loop
    # end index loop         
      
    return w, x

#-------------------------------------------------------------------------
# XINT - Numerical integration in the Fourier Domain using Ierly's method
#-------------------------------------------------------------------------

def xint(a,b,tol,vn,npts):
    """
    Quadrature by Ierley's method of Chebychev sampling.
    
    *Parameters*

    a : float
        upper limit of integration
    b : float
        upper limit of integration
    tol : float
        tolerance for integration
    vn : ndarray
        taper or Slepian sequence to convert-integrate
    npts : int
        number of points of tapers

    *Notes*

    This is a slight variation of Gleen Ierly's code. What was
    mainly done, was to avoid use of common blocks, defining all
    variables and performing the numerical integration inside
    (previously done by function pssevf).
    
    Exponential convergence rate for analytic functions!  Much faster 
    than Romberg; competitive with Gauss integration, without awkward 
    weights.
     
    Integrates the function dpsw on (a, b) to absolute
    accuracy tol > 0.
    
    the function in time is given by rpar with ipar points
      
    I removed the optional printing routine part of the code, 
    to make it easier to read. I also moved both nval, etol
    as normal variables inside the routine.
    
    nval = number of function calls made by routine
    etol = approximate magnitude of the error of the result
     
    NB:  function set_xint is called once before  xint  to
         provide quadrature samples and weights.
    
         I also altered the subroutine call, to get the weights
         and not save them in a common block, but get them 
         directly back.
    
    lomx=number of samples = 2**nomx
   
    *Modified*
    
    November 2004 (German A. Prieto)

    *Calls*
    
    utils.set_xint
    

    |

    """

    pi    = np.pi
    tpi   = 2.0 * pi
    nomx  = 8
    lomx  = 256
    ising = 1
    w, x = set_xint(ising)

    #--------------------------- 
    #   Check tol
    #---------------------------

    if (tol <= 0.0):
        raise ValueError("In xint tol must be > 0 ", tol)

    est = np.zeros(nomx,dtype=float)
    fv  = np.zeros(lomx+1,dtype=float)

    n  = 1
    im = 2**(nomx+1)

    for index in range(1,nomx+1):
        n   = 2*n
        im  = int(im/2)
        im2 = int(im/2)
        if (index <= 1):
            for i in range(n+1):
	       # Bottom
               y      = a+(b-a)*x[im2*i]
               om     = tpi*y
               ct, st = sft(vn,om)
               f1     = ct*ct+st*st
               # Top
               y      = b-(b-a)*x[im2*i]
               om     = tpi*y
               ct, st = sft(vn,om)
               f2     = ct*ct+st*st
         
               fv[im2*i] = f1 + f2   
            # end i loop, index 1, 
        else:
            for i in range(1,n,2):
	       # Bottom
               y     = a+(b-a)*x[im2*i]
               om    = tpi*y
               ct,st =  sft(vn,om)
               f1    = ct*ct+st*st
	       # Top
               y      = b-(b-a)*x[im2*i]
               om     = tpi*y
               ct, st = sft(vn,om)
               f2     = ct*ct+st*st
 
               fv[im2*i]= f1 + f2
            # end i loop, index > 1  
        # end index 1, or more
      
        x_int = 0.00
        for i in range(n+1):
            x_int = x_int + w[index-1, i]*fv[im2*i]
        x_int = x_int*(b-a)
        est[index-1] = x_int
        etol = 0.0

        # 
        #   Check for convergence. 
        #

        nval = 2*n
 
        if (index == 2):
            if ( est[index-1] == est[index-2] ):
                return x_int
           

        elif (index > 2):
            sq   = (est[index-1]-est[index-2])**2
            bot  = (0.01*sq + np.abs(est[index-1]-est[index-2]) )
            if (sq == 0.0):
                etol = 0.0
            else:
                etol = sq/bot
            if (etol <= tol): 
                return x_int
            
        # end check convergence
 
    # end index loop

    print('******** WARNING *********')
    print(' xint unable to provide requested accuracy')
    return x_int 


#-------------------------------------------------------------------------
# end XINT
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# DPSS_EV - Eigenvalues of the DPSS sequences
#-------------------------------------------------------------------------

def dpss_ev(vn,w,atol=1e-14):
    """
    Recalculate the DPSS eigenvalues, performing the 
    integration in the -W:W range, using Quadrature. 

    computes eigenvalues for the discrete prolate spheroidal sequences
    in efn by integration of the corresponding squared discrete prolate
    spheroidal wavefunctions over the inner domain. Due to symmetry, we
    perform integration from zero to w. 

    We use Chebychev quadrature for the numerical integration. 
  
    *Parameters*

    vn : ndarray [npts,kspec]
        DPSS to calculate eigenvalues 
    w : float
        the bandwidth (= time-bandwidth product/ndata)
    atol : float, optional
        absolute error tolerance for the integration. this should
        be set to 10**-n, where n is the number of significant figures
        that can be be represented on the machine.
        default = 1e-14

    *Returns*
    
    lamb : ndarray [kspec]
        vector of length vn.shape[1], contains the eigenvalues

    *Modified*
    
	November 2004 (German A. Prieto)

    *Calls*
    
    xint

    |

    """

    npts  = np.shape(vn)[0]
    kspec = np.shape(vn)[1]

    lamb = np.zeros(kspec)
    for k in range(kspec):

      result = xint(0.0,w,atol,vn[:,k],npts)
      lamb[k] = 2.0*result

    return lamb 

#-------------------------------------------------------------------------
# end DPSS_EV
#-------------------------------------------------------------------------

def dpss(npts,nw,kspec=None):
    """
    Calculation of the Discrete Prolate Spheroidal Sequences, and 
    the correspondent eigenvalues. 

    - Slepian, D.     1978  Bell Sys Tech J v57 n5 1371-1430
    - Thomson, D. J.  1982  Proc IEEE v70 n9 1055-1096

    **Parameters**
    
    npts : int
        the number of points in the series
    nw : float
        the time-bandwidth product (number of Rayleigh bins)
    kspec : int
        Optional, the desired number of tapers default = 2*nw-1

    **Returns**
    
    v : ndarray (npts,kspec)
        the eigenvectors (tapers) are returned in v[npts,nev]
    lamb : ndarray (kspec) 	
        the eigenvalues of the v's

    **Notes**
    
    In SCIPY the codes are already available to calculate the DPSS. 
    Eigenvalues are calculated using Chebeshev Quadrature. 
    Code also performs interpolation if NPTS>1e5
    
    Also, define DPSS to be positive-standard, meaning vn's always 
    start positive, whether symmetric or not. 

    **Modified**
    
	December 2020

    **Calls**
    
    scipy.signal.windows.dpss
    dpss_ev

    |

    """

   
    #-----------------------------------------------------
    # Check number of tapers
    #-----------------------------------------------------

    W = nw/float(npts)

    if (kspec is None):
       kspec = np.int(np.round(2*nw-1))

    #-----------------------------------------------------
    # Get the DPSS, using SCIPY 
    #    Interpolate if necesary
    #-----------------------------------------------------

    if (npts < 1e5):
        v,lamb2 = signal.windows.dpss(npts, nw, Kmax=kspec, 
                            sym=True,norm=2,
                            return_ratios=True)

        v      = v.transpose()
    else:
        lsize = np.floor(np.log10(npts))
        nint  = int((10**lsize))
        print('DPSS using interpolation', npts, nint)
        v2int = signal.windows.dpss(nint, nw, Kmax=kspec, 
                            sym=True,norm=2)

        v2int = v2int.transpose()
        v     = np.zeros((npts,kspec),dtype=float)
        x      = np.arange(nint)
        y      = np.linspace(0,nint-1,npts,endpoint=True)
        for k in range(kspec):
            I      = interp.interp1d(x, v2int[:,k], kind='quadratic') 
                                                        #'quadratic')
            v[:,k] = I(y)
            v[:,k] = v[:,k]*np.sqrt(float(nint)/float(npts))

    #-----------------------------------------------------
    # Normmalize functions
    #-----------------------------------------------------

    for i in range(kspec):
        vnorm  = np.sqrt(np.sum(v[:,i]**2))
        v[:,i] = v[:,i]/vnorm

    #-----------------------------------------------------
    # Get positive standard
    #-----------------------------------------------------

    nx = npts%2
    if (nx==1):
        lh = int((npts+1)/2)
    else:
        lh = int(npts/2)

    for i in range(kspec):
        if (v[lh,i] < 0.0):
           v[:,i] = -v[:,i]

    lamb = dpss_ev(v,W)

    return v, lamb

#-------------------------------------------------------------------------
# end DPSS 
#-------------------------------------------------------------------------

def dpss2(npts,nw,nev=None):
    """
    This is a try to compute the DPSS using the original Thomson
    approach. It reduces the problem to half the size and inverts
    independently for the even and odd functions. 
    
    This is work in progress and not used. 

    Modified from F90 library:
	German Prieto
	December 2020

    The tapers are the eigenvectors of the tridiagonal matrix sigma(i,j)
    [see Slepian(1978) eq 14 and 25.] They are also the eigenvectors of
    the Toeplitz matrix eq. 18. We solve the tridiagonal system in

    scipy.linalg.eigh_tridiagonal
    
    (real symmetric tridiagonal solver) for the tapers and use 
    them in the integral equation in the frequency domain 
    (dpss_ev subroutine) to get the eigenvalues more accurately, 
    by performing Chebychev Gaussian Quadrature following Thomson's codes.
   
    First, we create the main and off-diagonal vectors of the 
    tridiagonal matrix. We compute separetely the even and odd tapers, 
    by calling eigh_tridiagonal from SCIPY.
    
    We, refine the eigenvalues, by computing the inner bandwidth 
    energy in the frequency domain (eq. 2.6 Thomson). Also the "leakage"
    (1 - eigenvalue) is estimated, independenly if necesary. 
   

    In SCIPY the codea are already available to calculate the DPSS. 
    Eigenvalues are calculated using Chebeshev Quadrature. 

    Code also performs interpolation if NPTS>1e5
    Also, define DPSS to be positive-standard, meaning vn's always 
    start positive, whether symmetric or not. 

    **Calls**
    
    To do

    |

    """

    #-----------------------------------------------------
    # Check number of tapers
    #-----------------------------------------------------

    bw = nw/float(npts)

    if (nev is None):
       nev = np.int(np.round(2*nw-1))

    #-----------------------------------------------------
    # Check size of vectors and half lengths
    #-----------------------------------------------------
 
    nx = npts%2
    if (nx==1):
        lh = int((npts+1)/2)
    else:
        lh = int(npts/2)

    nodd  = int ((nev-(nev%2))/2)
    neven = nev - nodd

    com = np.cos(2.0*np.pi*bw)
    hn  = float(npts-1.0)/2.0
    r2  = np.sqrt(2.0)

    # Initiate eigenvalues and eigenvectors
    v     = np.zeros((npts,nev),dtype=float)
    theta = np.zeros(nev,dtype=float)

    #---------------------------------------------
    # Do even tapers
    #---------------------------------------------

    fv1 = np.zeros(lh,dtype=float)
    fv2 = np.zeros(lh,dtype=float)

    for i in range(lh):
        n = i
        fv1[i]   = com*(hn - float(n))**2.0
        fv2[i]   = float(n*(npts-n))/2.0

    if (nx == 0):
        fv1[lh-1] = com*(hn-float(lh-1))**2.0 + float(lh*(npts-lh))/2.0
    else:
        fv2[lh-1] = r2*fv2[lh-1]

    fv3 = fv2[1:lh]
    eigval,v2 = linalg.eigh_tridiagonal(fv1, fv2[1:lh],
                     select='i',select_range=(lh-neven,lh-1))

    if (nx==1):
        for k in range(neven):
            v[lh,k] = v[lh,k]*r2

    for k in range(neven):
        kr = k
        k2 = 2*k

        theta[k2] = eigval[kr]

        nr = npts-1
        for i in range(lh):
           v[i,k2]  = v2[i,kr]
           v[nr,k2] = v2[i,kr]
           nr=nr-1
    
    #---------------------------------------------
    # Do odd tapers
    #---------------------------------------------

    fv1 = np.zeros(lh,dtype=float)
    fv2 = np.zeros(lh,dtype=float)

    if (nodd > 0):   

      for i in range(lh):
         n = i
         fv1[i]  = com*(hn - float(n))**2
         fv2[i]  = float(n*(npts-n))/2.0
   
      if (nx == 0):
         fv1[lh-1] = com*(hn-float(lh-1))**2 - float(lh*(npts-lh))/2.0
   
      eigval,v2 = linalg.eigh_tridiagonal(fv1, fv2[1:lh],
                     select='i',select_range=(lh-nodd,lh-1))

      for k in range(nodd):
          kr = k
          k2 = 2*k+1 
          
          theta[k2] = eigval[kr]

          nr = npts-1
          for i in range(lh):
              v[i,k2]  = v2[i,kr]
              v[nr,k2] = -v2[i,kr]
              nr=nr-1

    #---------------------------------------
    # Normalize the eigenfunction
    # and positive standard
    #---------------------------------------

    for i in range(nev):
        vnorm  = np.sqrt(np.sum(v[:,i]**2))
        v[:,i] = v[:,i]/vnorm
        if (v[lh,i]<0.0):
            v[:,i] = -v[:,i]


    v = np.flip(v,axis=1)
    lamb = dpss_ev(v,bw)

    return v, lamb

#-------------------------------------------------------------------------
# end DPSS - my version 
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
# Eigenspec
#-------------------------------------------------------------------------

def eigenspec(x,vn,lamb,nfft):
    """
    Calculate eigenspectra using DPSS sequences. 
    Gets yk's from Thomson (1982). 

    **Parameters**
    
    x : ndarray [npts,0]
        real vector with the time series
    vn : ndarray [npts,kspec]	
        the different tapers computed in dpss
    lambda : ndarray [kspec]	
        the eigenvalues of the tapers vn
    nfft : int		
        number of frequency points (inc. positive 
        and negative frequencies)

    **Returns**
    
    yk : complex ndarray [kspec,nfft]	
        complex array with kspec fft's of tapered 
        data. Regardless of real/complex input data
        all frequencies are stored. Good for coherence, 
        deconvolution, etc. 
    sk : ndarray [kspec,nfft]  
        real array with kspec eigenspectra

    **Modified**
    
 	German Prieto
	November 2004

    **Notes**
    
    Computes eigen-ft's by windowing real data with dpss and taking ffts
    Note that fft is unnormalized and window is such that its sum of 
    squares is one, so that psd=yk**2.
    
    The fft's are computed using SCIPY FFT codes, and parallel FFT can 
    potentially speed up the calculation. Up to KSPEC works are sent.
    The yk's are saved to get phase information. Note that tapers are 
    applied to the original data (npts long) and the FFT is zero padded
    up to NFFT points. 
 
    **Calls**
    
    scipy.fft.fft

    |

    """

    kspec = np.shape(vn)[1]
    npts  = np.shape(x)[0]

    if (nfft < npts):
        raise ValueError("NFFT must be larger than NPTS ", npts, nfft)

    k2 = vn.shape[1]
    if (kspec > k2):
        raise ValueError("DPSS dimensions don't agree ", kspec, k2, ' tapers')

    #-----------------------------------------------------------------
    # Define matrices to be used
    #-----------------------------------------------------------------

    xtap   = np.zeros((npts,kspec), dtype=float)
    for i in range(kspec):
        xtap[:,i]     = vn[:,i]*x[:,0]
    yk  = scipy.fft.fft(xtap,axis=0,n=nfft,workers=kspec)
    sk  = np.abs(yk)**2

    return yk, sk


#-------------------------------------------------------------------------
# end Eigenspec
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Adaptspec
#-------------------------------------------------------------------------

def adaptspec(yk,sk,lamb,iadapt=0):
    """
    Calculate adaptively weighted power spectrum
    Options for non-adaptive estimates are posible, with optional parameter 
    iadapt, using average of sk's or weighted by eigenvalue. 

    **Parameters**
    
    yk : complex ndarray [nfft,kspec]    
        complex array of kspec eigencoefficients 
    sk : ndarray [nfft,kspec]    
        array containing kspe power spectra
    lamb : ndarray [kspec]
        eigenvalues of tapers
    iadapt : int
        defines methos to use, default = 0
        0 - adaptive multitaper
        1 - unweighted, wt =1 for all tapers
        2 - wt by the eigenvalue of DPSS


    **Returns**
    
    spec : ndarray [nfft]
        real vector containing adaptively weighted spectrum
    se : ndarray [nfft]
        real vector containing the number of degrees of freedom
        for the spectral estimate at each frequency.
    wt : ndarray [nfft,kspec]
        real array containing the ne weights for kspec 
        eigenspectra normalized so that if there is no bias, the
        weights are unity.
 
    **Modified**
    
 	German Prieto, Aug 2006

    Corrected the estimation of the dofs se (sum of squares of wt is 1.0)
    maximum wt = 1

    German Prieto, October 2007	
    Added the an additional subroutine noadaptspec to calculate a simple non-adaptive multitaper spectrum.
    This can be used in transfer functions and deconvolution, 
    where adaptive methods might not be necesary. 

    **Calls**
    
    nothing

    |

    """

    mloop = 1000
    nfft  = np.shape(yk)[0]
    kspec = np.shape(yk)[1]
    lamb1 = 1.0-lamb

    sbar = np.zeros((nfft,1),     dtype=float)
    se   = np.zeros((nfft,1),     dtype=float)
    wt   = np.zeros((nfft,kspec), dtype=float)
    skw  = np.zeros((nfft,kspec), dtype=float)

    #----------------------------------------------------
    # Simple average, not adaptive. Weight=1
    #    iadapt=1
    #----------------------------------------------------

    if (iadapt==1):
        wt = wt + 1.0
        sbar[:,0] = np.sum(sk,axis=1)/ float(kspec)
        se        = se + 2.0 * float(kspec)
        spec      = sbar 
        return spec, se, wt

    #----------------------------------------------------
    # Weight by eigenvalue of Slepian functions
    #    iadapt=2
    #----------------------------------------------------


    if (iadapt==2):
        for k in range(kspec):
            wt[:,k]  = lamb[k]
            skw[:,k] = wt[:,k]**2 * sk[:,k]   

        wtsum     = np.sum(wt**2,axis=1)
        skwsum    = np.sum(skw,axis=1)
        sbar[:,0] = skwsum / wtsum
        spec      = sbar 

        #------------------------------------------------------------
        # Number of Degrees of freedom
        #------------------------------------------------------------

        se = wt2dof(wt)
        
        return spec, se, wt

    #----------------------------------------
    # Freq sampling (assume unit sampling)
    #----------------------------------------
    df = 1.0/float(nfft-1)

    #----------------------------------------
    # Variance of Sk's and avg variance
    #----------------------------------------

    varsk  = np.sum(sk,axis=0)*df
    dvar   = np.mean(varsk)

    bk	  = dvar  * lamb1  # Eq 5.1b Thomson
    sqlamb = np.sqrt(lamb)

    #-------------------------------------------------
    # Iterate to find optimal spectrum
    #-------------------------------------------------

    rerr = 9.5e-7	# Value used in F90 codes check

    sbar[:,0] = (sk[:,0] + sk[:,1])/2.0
    spec = sbar
    
    for i in range(mloop):
    
        slast = np.copy(sbar)

        for k in range(kspec):
            wt[:,k]  = sqlamb[k]*sbar[:,0] /(lamb[k]*sbar[:,0] + bk[k])
            wt[:,k]  = np.minimum(wt[:,k],1.0)
            skw[:,k] = wt[:,k]**2 * sk[:,k]   

        wtsum     = np.sum(wt**2,axis=1)
        skwsum    = np.sum(skw,axis=1)
        sbar[:,0] = skwsum / wtsum

        oerr = np.max(np.abs((sbar-slast)/(sbar+slast)))

        if (i==mloop): 
            spec = sbar
            print('adaptspec did not converge, rerr = ',oerr, rerr)
            break
        
        if (oerr > rerr):
            continue

        spec = sbar
        break

    spec = sbar
    #---------

    #------------------------------------------------------------
    # Number of Degrees of freedom
    #------------------------------------------------------------
  
    se = wt2dof(wt)
 
    return spec, se, wt


#-------------------------------------------------------------------------
# end adaptspec 
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# jackspec
#-------------------------------------------------------------------------

def jackspec(spec,sk,wt,se):

    """
    code to calculate adaptively weighted jackknifed 95% confidence limits

    **Parameters**
    
    spec : ndarray [nfft]
        real vector containing adaptively weighted spectrum
    sk : ndarray [nfft,kspec]
        array with kth power spectra 
    wt : ndarray [nfft,kspec]
        real array containing the ne weights for kspec 
        eigenspectra normalized so that if there is no bias, the
        weights are unity.
    se : ndarray [nfft]
        real vector containing the number of degrees of freedom
        for the spectral estimate at each frequency.
     
    **Returns**

    spec_ci : ndarray [nfft,2]
        real array of jackknife error estimates, with 5 and 95%
        confidence intervals of the spectrum.

    **Calls**
    
    scipy.stats.t.ppf

    **Modified**
    
    German Prieto, Aug 2006
	
    German Prieto, March 2007
    
    Changed the Jackknife to be more efficient.
    
    |

    """
    #------------------------------------------------------
    # Get sizes and define matrices
    #------------------------------------------------------

    nfft    = np.shape(sk)[0]
    kspec   = np.shape(sk)[1]
    wjk     = np.zeros((nfft,kspec-1))
    sj      = np.zeros((nfft,kspec-1))
    sjk     = np.zeros((nfft,kspec))
    varjk   = np.zeros((nfft,kspec))
    var     = np.zeros((nfft,1))

    #------------------------------------------------------
    # Do simple jackknife
    #------------------------------------------------------

    for i in range(kspec):
        ks = -1 
        for k in range(kspec):

            if (k == i):
                continue
            ks = ks + 1
 
            wjk[:,ks] = wt[:,k]
            sj[:,ks]  = wjk[:,ks]**2 * sk[:,k]   
      
        sjk[:,i] = np.sum(sj,axis=1)/ np.sum(wjk**2,axis=1) 

    #------------------------------------------------------
    # Jackknife mean (Log S)
    #------------------------------------------------------

    lspec = np.log(spec)
    lsjk  = np.log(sjk)

    lsjk_mean = np.sum(lsjk, axis=1)/float(kspec)

    #------------------------------------------------------
    # Jackknife Bias estimate (Log S)
    #------------------------------------------------------

    bjk = float(kspec-1) * (lspec - lsjk_mean)

    #------------------------------------------------------
    # Jackknife Variance estimate (Log S)
    #------------------------------------------------------

    for i in range(kspec):
        varjk[:,i] = (lsjk[:,i] - lsjk_mean)**2
    var[:,0] = np.sum(varjk, axis=1) * float(kspec-1)/float(kspec)

    #------------------------------------------------------
    # Use the degrees of freedom
    #------------------------------------------------------

    for i in range(nfft):
        if (se[i]<1.0):
            print('DOF < 1 ', i,'th frequency ', se[i])
            raise ValueError("Jackknife - DOF are wrong")
        qt = scipy.stats.t(df=se[i]).ppf((0.95))
        var[i,0] = np.exp(qt)*np.sqrt(var[i,0])

    #-----------------------------------------------------------------
    # Clear variables
    #-----------------------------------------------------------------

    del wjk, sj, sjk, varjk

    #-----------------------------------------------------------------
    # Return confidence intervals
    #-----------------------------------------------------------------


    spec_ci  = np.zeros((nfft,2))
    ci_dw    = spec/var
    ci_up    = spec*var

    spec_ci[:,0]  = ci_dw[:,0]
    spec_ci[:,1]  = ci_up[:,0]

    return spec_ci 

#-------------------------------------------------------------------------
# end jackspec
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
# qiinv
#-------------------------------------------------------------------------

def qiinv(spec,yk,wt,vn,lamb,nw):

    """
    Function to calculate the Quadratic Spectrum using the method 
    developed by Prieto et al. (2007).   
    
    The first 2 derivatives of the spectrum are estimated and the 
    bias associated with curvature (2nd derivative) is reduced. 

    Calculate the Stationary Inverse Theory Spectrum.
    Basically, compute the spectrum inside the innerband. 
  
    This approach is very similar to D.J. Thomson (1990).

    **Parameters**
    
    spec : ndarray [nfft,0]    
        the adaptive multitaper spectrum (so far)
    yk : ndarrau, complex [npts,kspec]  
        multitaper eigencoefficients, complex
    wt : ndarray [nf,kspec]	
        the weights of the different coefficients. 
        input is the original multitaper weights, 
        from the Thomson adaptive weighting. 
  	vn : ndarray [npts,kspec]  
        the Slepian sequences
    lambda : ndarray [kspec]   
        the eigenvalues of the Slepian sequences
    nw : float             
        The time-bandwisth product

    **Returns**
    
    qispec : ndarray [nfft,0]
        the QI spectrum estimate
    ds : ndarray [nfft,0]	
        the estimate of the first derivative
    dds : ndarray [nfft,0]	
        the estimate of the second derivative

    **References**
    
    G. A. Prieto, R. L. Parker, D. J. Thomson, F. L. Vernon, 
    and R. L. Graham (2007), Reducing the bias of multitaper 
    spectrum estimates,  Geophys. J. Int., 171, 1269-1281. 
    doi: 10.1111/j.1365-246X.2007.03592.x.
   
    **Notes**
    
    In here I have made the Chebyshev polinomials unitless, 
    meaning that the associated parameters ALL have units 
    of the PSD and need to be normalized by 1/W for \alpha_1, 
    1/W**2 for \alpha_2, etc.

    **Modified**
    
    Nov 2021 (German A Prieto)
    
    Major adjustment in the inverse problem steps. 
    Now, the constant term is first inverted for, 
    and then the 1st and 2nd derivative so that we 
    obtain an independent 2nd derivative.
    
    June 5, 2009 (German A. Prieto)
  	
    Major change, saving some important
    values so that if the subroutine is called 
    more than once, with similar values, many of 
    the variables are not calculated again, making 
    the code run much faster. 

    **Calls**
    
    scipy.optimize.nnls, scipy.linalg.qr,
    scipy.linalg.lstsq

    |

    """

    npts  = np.shape(vn)[0] 
    kspec = np.shape(vn)[1]
    nfft  = np.shape(yk)[0]
    nfft2 = 11*nfft
    nxi   = 79;
    L     = kspec*kspec;

    if (np.min(lamb) < 0.9): 
        print('Careful, Poor leakage of eigenvalue ', np.min(lamb));
        print('Value of kspec is too large, revise? *****') 

    #---------------------------------------------
    # Assign matrices to memory
    #---------------------------------------------

    xk     = np.zeros((nfft,kspec), dtype=complex)
    Vj     = np.zeros((nxi,kspec),  dtype=complex)

    #---------------------------------------
    # New inner bandwidth frequency
    #---------------------------------------

    bp   = nw/npts		# W bandwidth
    xi   = np.linspace(-bp,bp,num=nxi)
    dxi  = xi[2]-xi[1]
    f_qi = scipy.fft.fftfreq(nfft2)

    for k in range(kspec):
        xk[:,k] = wt[:,k]*yk[:,k];
        for i in range(nxi):
            om = 2.0*np.pi*xi[i]
            ct,st = sft(vn[:,k],om) 
            Vj[i,k] = 1.0/np.sqrt(lamb[k])*complex(ct,st)

    #----------------------------------------------------------------
    # Create the vectorized Cjk matrix and Pjk matrix { Vj Vk* }
    #----------------------------------------------------------------
   
    C      = np.zeros((L,nfft),dtype=complex)
    Pk     = np.zeros((L,nxi), dtype=complex)
 
    m = -1;
    for i in range(kspec):
        for k in range(kspec):
            m = m + 1;
            C[m,:] = ( np.conjugate(xk[:,i]) * (xk[:,k]) );

            Pk[m,:] = np.conjugate(Vj[:,i]) * (Vj[:,k]);

    Pk[:,0]         = 0.5 * Pk[:,0];
    Pk[:,nxi-1]     = 0.5 * Pk[:,nxi-1];

    #-----------------------------------------------------------
    # I use the Chebyshev Polynomial as the expansion basis.
    #-----------------------------------------------------------

    hk     = np.zeros((L,3),   dtype=complex)
    hcte   = np.ones((nxi,1),  dtype=float)
    hslope = np.zeros((nxi,1), dtype=float)
    hquad  = np.zeros((nxi,1), dtype=float)
    Cjk    = np.zeros((L,1),   dtype=complex)
    cte    = np.zeros(nfft)
    cte2   = np.zeros(nfft)
    slope  = np.zeros(nfft)
    quad   = np.zeros(nfft)
    sigma2 = np.zeros(nfft)
    cte_var    = np.zeros(nfft)
    slope_var  = np.zeros(nfft)
    quad_var   = np.zeros(nfft)

    h1 = np.matmul(Pk,hcte) * dxi
    hk[:,0] = h1[:,0]

    hslope[:,0] = xi/bp 
    h2 = np.matmul(Pk,hslope) * dxi 
    hk[:,1] = h2[:,0]

    hquad[:,0] = (2.0*((xi/bp)**2) - 1.0)
    h3 = np.matmul(Pk,hquad) * dxi
    hk[:,2] = h3[:,0]
    nh = np.shape(hk)[1]

    #----------------------------------------------------
    # Begin Least squares solution (QR factorization)
    #----------------------------------------------------

    Q,R  = scipy.linalg.qr(hk);
    Qt   = np.transpose(Q)
    Leye = np.eye(L)
    Ri,res,rnk,s = scipy.linalg.lstsq(R,Leye)
    covb = np.real(np.matmul(Ri,np.transpose(Ri))) 

    for i in range(nfft):
       Cjk[:,0]    = C[:,i]
#       hmodel,res,rnk,s = scipy.linalg.lstsq(hk,Cjk)
       btilde    = np.matmul(Qt,Cjk) 
       hmodel,res,rnk,s = scipy.linalg.lstsq(R,btilde)

       #---------------------------------------------
       # Estimate positive spectrumm
       #---------------------------------------------
       cte_out  = optim.nnls(np.real(h1), 
                             np.real(Cjk[:,0]))[0]
       cte2[i]  = np.real(cte_out) 
       pred = h1*cte2[i]
       Cjk2 = Cjk-pred
       #---------------------------------------------
       # Now, solve the derivatives
       #---------------------------------------------
       btilde    = np.matmul(Qt,Cjk2) 
       hmodel,res,rnk,s = scipy.linalg.lstsq(R,btilde)
       cte[i]   = np.real(hmodel[0])
       slope[i] = -np.real(hmodel[1])
       quad[i]  = np.real(hmodel[2])

       pred = np.matmul(hk,np.real(hmodel))
       sigma2[i] = np.sum(np.abs(Cjk-pred)**2)/(L-nh) 

       cte_var[i]   = sigma2[i]*covb[0,0]
       slope_var[i] = sigma2[i]*covb[1,1]
       quad_var[i]  = sigma2[i]*covb[2,2]

    slope = slope / (bp)
    quad  = quad  / (bp**2)

    slope_var = slope_var / (bp**2)
    quad_var = quad_var / (bp**4)

    qispec = np.zeros((nfft,1), dtype=float)    
    for i in range(nfft):
        qicorr = (quad[i]**2)/((quad[i]**2) + quad_var[i] ) 
        qicorr = qicorr * (1/6)*(bp**2)*quad[i]

        qispec[i] = cte2[i] - qicorr
        #qispec[i] = spec[i] - qicorr


    ds  = slope;
    dds = quad;

    ds  = ds[:,np.newaxis]
    dds = dds[:,np.newaxis]

    return qispec, ds, dds

#-------------------------------------------------------------------------
# end qiinv
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# ftest
#-------------------------------------------------------------------------

def ftest(vn,yk):
    """
    Performs the F test for a line component
  
    Compute F-test for single spectral line components
    at the frequency bins given by the mtspec routines. 
    
    **Parameters**
    
    vn : ndarray [npts,kspec]
        Slepian sequences real
    yk : ndarray, complex [nfft,kspec]
        multitaper eigencoefficients, complex
        kspec fft's of tapered data series 
  
    **Returns**
    
  	F : ndarray [nfft]
        vector of f-test values, real
  	p : ndarray [nfft]
        vector with probability of line component

    **Calls** 
    
        scipy.stats.f.cdf, scipy.stats.f.cdf

    |

    """

    npts  = np.shape(vn)[0]
    kspec = np.shape(vn)[1]
    nfft  = np.shape(yk)[0]
    mu    = np.zeros(nfft,dtype=complex)
    F     = np.zeros(nfft)
    p     = np.zeros(nfft)

    dof1 = 2
    dof2 = 2*(kspec-1)

    #------------------------------------------------------
    #  The Vk(0), summing the time domain tapers
    #  Also normalize by sum(vn0)**2
    #------------------------------------------------------

    vn0 = np.sum(vn,axis=0)
    vn0_sqsum = np.sum(np.abs(vn0)**2)
    
    #------------------------------------------------------
    #  Calculate the mean amplitude of line components at 
    #  each frequency
    #------------------------------------------------------

    for i in range(nfft):
        vn_yk     = vn0[:]*yk[i,:]
        vn_yk_sum = np.sum(vn_yk)
        mu[i]     = vn_yk_sum/vn0_sqsum

    #------------------------------------------------------
    # Calculate F Test
    #    Top	(kspec-1) mu**2 sum(vn0**2)  Model variance
    #    Bottom	sum(yk - mu*vn0)**2	     Misfit
    # Fcrit - IS the threshhold for 95% test.
    #------------------------------------------------------
   
    Fcrit = scipy.stats.f.ppf(0.95,dof1,dof2) 
    for i in range(nfft):
        Fup  = float(kspec-1) * np.abs(mu[i])**2 * np.sum(vn0**2)
        Fdw  = np.sum( np.abs(yk[i,:] - mu[i]*vn0[:])**2 )
        F[i] = Fup/Fdw
        p[i] = scipy.stats.f.cdf(F[i],dof1,dof2)

    F = F[:,np.newaxis]
    p = p[:,np.newaxis]

    return F, p

#-------------------------------------------------------------------------
# end ftest 
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# reshape spectrum
#-------------------------------------------------------------------------

def yk_reshape(yk_in,vn,p=None,fcrit=0.95):  
    """
    reshape the yk's based on the F-test of line compenents
  
    Reshape eigenft's around significant spectral lines
    The "significant" means above fcritical probability (def=0.95)
    If probability is large at neighbouring frequencies, code will 
    only remove the largest probability energy. 
  
    **Parameters**
    
    yk : ndarray complex [nfft,kspec] 
        eigenft's
    vn : ndarray [npts,kspec] 
        DPSS sequences
    p : ndarray optional [nfft] 
        F-test probabilities to find fcritical
        In None, it will be calculated
    fcrit : float optional
        Probability value over which to reshape, default = 0.95

    **Returns**
    
    yk : ndarray, complex [nfft,kspec]
        Reshaped eigenft's 
    sline : ndarray [nfft]
        Power spetrum of line components only

    **Modified**
  	
  	April 2006 (German A. Prieto)

    **Calls**
    
       ftest - if P is not present
       scipy.fft.fft

    |

    """

    if (p is None):
        print('Doing F test')
        p        = utils.ftest(vn,yk)[1]
    yk       = np.copy(yk_in)
    npts     = np.shape(vn)[0]
    kspec    = np.shape(vn)[1]
    nfft     = np.shape(yk)[0]
    sline    = np.zeros((nfft,1),dtype=float)
    Vk       = np.zeros((nfft,kspec),dtype=complex)

    #------------------------------------------------------
    # Count and isolate, peaks that pass
    # the fcrit criteria. 
    #    Also, remove values which are not local peaks
    #------------------------------------------------------

    nl = 0
    for i in range(nfft):
        if (p[i] < fcrit):
            p[i] = 0
            continue

        if (i==0):
            if (p[i]>p[i+1]):
                nl = nl + 1
            else:
                p[i] = 0.0
        elif (i==nfft-1):
            if (p[i]>p[i-1]):
                nl = nl + 1
            else:
                p[i] = 0
        else:
            if (p[i]>p[i-1] and p[i]>p[i+1]):
                nl = nl + 1
            else:
                p[i] = 0

    #------------------------------------------------------
    # If no lines are found, return back arrays
    #------------------------------------------------------

    if (nl == 0): 
        return yk,sline

    #------------------------------------------------------
    # Prepare vn's Vk's for line removal
    #    Compute the Vk's to reshape
    #    The Vk's normalized to have int -1/2 1/2 Vk**2 = 1 
    #    This is obtained from fft already is sum(vn**2) = 1
    #------------------------------------------------------

    vn0 = np.sum(vn,axis=0)

    for k in range(kspec):
        Vk[:,k] = scipy.fft.fft(vn[:,k],nfft)
    
    #------------------------------------------------------
    #  Remove mean value for each spectral line
    #------------------------------------------------------

    for i in range(nfft):
        if (p[i]<fcrit):
            continue
        mu = np.sum(vn0*yk[i,:]) / np.sum(vn0**2)

        for j in range(nfft):
            jj = j - i 
            if (jj < 0):
                jj = jj + nfft

            yk_pred = mu*Vk[jj,:]
            yk[j,:] = yk[j,:] - yk_pred
            #yk[j,:] = yk[j,:] - mu*Vk[jj,:]

            for k in range(kspec):
                kfloat = 1.0/float(kspec)
                sline[i] = sline[i] + kfloat*np.abs(mu*Vk[jj,k])**2


    return yk, sline

#-------------------------------------------------------------------------
# end reshape 
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Calculate degrees of freedom
#-------------------------------------------------------------------------

def wt2dof(wt):
    """
    Calculate the degrees of freedom of the multitaper based on the 
    weights of the different tapers.

    **Parameters**
    
    wt : ndarray [nfft,kspec] 
        weights of the tapers at each frequency

    **Returns**
    
    se : ndarray [nfft] 
        degrees of freedom at each frequency

    | 

    """
    
    nfft  = np.shape(wt)[0]
    kspec = np.shape(wt)[1]
 
    #------------------------------------------------------------
    # Number of Degrees of freedom
    #------------------------------------------------------------

    wt_dofs   = np.zeros((nfft,kspec), dtype=float)
    for i in range(nfft):
        wt_dofs[i,:] = wt[i,:]/np.sqrt(np.sum(wt[i,:]**2)/float(kspec))
    wt_dofs = np.minimum(wt_dofs,1.0)
    
    se = 2.0 * np.sum(wt_dofs**2, axis=1) 

    return se

#-------------------------------------------------------------------------
# End DOFs
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Dual-frequency spectrum
#-------------------------------------------------------------------------

def df_spec(x,y=None,fmin=None,fmax=None):
    """
    Dual frequency spectrum using one/two MTSPEC classes. 
    For now, only positive frequencies are studied
  
    Construct the dual-frequency spectrum from the yk's and the 
    weights of the usual multitaper spectrum estimation. 
  
    **Parameters**
    
    x : MTSpec class
        variable with the multitaper information (yk's)
    y : MTSpec class, optional
        similar to x for a second time series
        if y is None, auto-dual frequency is calculated.
    fmin : float, optional
        minimum frequency to calculate the DF spectrum
    fmax : float, optional
        minimum frequency to calculate the DF spectrum

  
    **Returns**
    
    df_spec : ndarray complex, 2D (nf,nf)
        the complex dual-frequency cross-spectrum. Not normalized
    df_cohe : ndarray, 2D (nf,nf)
        MSC, dual-freq coherence matrix. Normalized (0.0,1.0)
    df_phase : ndarray, 2D (nf,nf)
        the dual-frequency phase

    **Notes**
    
    both x and y need the same parameters (npts, kspec, etc.)

    **Modified**
    
  	German Prieto, September 2005
  
  	German A. Prieto, September 2007
  	
    Slight rewrite to adjust to newer mtspec codes.
    
    **Calls**
    
  	Nothing

    |

    """

    if (y is None):
        y = x

    kspec = x.kspec
    nfft  = x.nfft
    nf    = x.nf
    freq  = x.freq[:,0]
    if (fmin is None):
        fmin = min(abs(freq))
    if (fmax is None):
        fmax = max(abs(freq))
    floc = np.zeros(nf,dtype=int)
    icnt = -1
    for i in range(nf):
        if (freq[i]>=fmin and freq[i]<=fmax):
            icnt = icnt + 1
            floc[icnt] = i
    floc = floc[0:icnt]
    nf   = icnt
    freq = freq[floc]

    #------------------------------------------------------------
    # Create the cross and/or auto spectra
    #------------------------------------------------------------

    # Unique weights (and degrees of freedom)
    wt = np.minimum(x.wt,y.wt)

    wt_scale = np.sum(np.abs(wt)**2, axis=1)  # Scale weights to keep power 
    for k in range(kspec):
         wt[:,k] = wt[:,k]/np.sqrt(wt_scale)

    # Weighted Yk's
    dyk_x = np.zeros((nf,kspec),dtype=complex)
    dyk_y = np.zeros((nf,kspec),dtype=complex)
    for k in range(kspec):
        dyk_x[:,k] = wt[floc,k] * x.yk[floc,k]
        dyk_y[:,k] = wt[floc,k] * y.yk[floc,k]


    # Auto and Cross spectrum
    Sxx      = np.zeros((nf,1),dtype=float)
    Syy      = np.zeros((nf,1),dtype=float)
    Sxx[:,0] = np.sum(np.abs(dyk_x)**2, axis=1) 
    Syy[:,0] = np.sum(np.abs(dyk_y)**2, axis=1)
 
    # Get coherence and phase
    df_spec  = np.zeros((nf,nf),dtype=complex)
    df_cohe  = np.zeros((nf,nf),dtype=float)
    df_phase = np.zeros((nf,nf),dtype=float)
 
    for i in range(nf):

        if ((i+1)%1000==0):
            print('DF_SPEC ith loop ',i+1,' of ',nf)
        for j in range(nf):
            df_spec[i,j]  = np.sum(dyk_x[i,:] * np.conjugate(dyk_y[j,:]))
            df_cohe[i,j]  = np.abs(df_spec[i,j])**2 / (Sxx[i]*Syy[j])
            df_phase[i,j] = np.arctan2( np.imag(df_spec[i,j]),
                                         np.real(df_spec[i,j]) ) 

    df_phase = df_phase * (180.0/np.pi)

    return df_spec, df_cohe, df_phase, freq

#-------------------------------------------------------------------------
# End DF_SPEC
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# SFT - slow fourier transform
#-------------------------------------------------------------------------

def sft(x,om):
    """
    calculates the (slow) fourier transform of real 
    sequence x(i),i=1,...n at angular frequency om normalized 
    so that nyquist=pi. the sine transform is returned in st and 
    the cosine transform in ct.
    
    algorithm is that of goertzal with modifications by
    gentleman, comp.j. 1969
    
    transform is not normalized
    
    to normalize one-sided ft, divide by sqrt(data length)
    for positive om, the ft is defined as ct-(0.,1.)st or like slatec
    cfftf

    **Parameters**
    
    x : ndarray (n,) 
        time sequence x[0],x[1],...
    om : float
        angular frequency of interest,
        normalized such that Nyq = pi

    **Modified**
    
	German Prieto
	November 2004

    |

    """ 

    n = np.shape(x)[0]
    pi = np.pi
    tp = 2.0*pi

    np1 = n+1
    l   = int(np.floor(6.0*om/tp))
    s   = np.sin(om)
    a   = 0.0
    c   = 0.0
    d   = 0.0
    e   = 0.0

    if (l == 0):

        # recursion for low frequencies (.lt. nyq/3)

        b = -4.0*np.sin(om/2.0)**2
        for k0 in range(n):
            k = k0+1
            c = a
            d = e
            a = x[np1-k-1]+b*d+c
            e = a+d
      
    elif (l == 1):

        #regular goertzal algorithm for intermediate frequencies

        b = 2.0*np.cos(om)
        for k0 in range(n):
            k = k0 + 1
            a = x[np1-k-1]+b*e-d
            d = e
            e = a

    else:
        # recursion for high frequencies (> 2*fnyq/3)
      
        b=4.0*np.cos(om/2.0)**2
        for k0 in range(n):
            k = k0 + 1
            c = a
            d = e
            a = x[np1-k-1]+b*d-c
            e = a-d

    st = -s*d
    ct = a-b*d/2.0

    return ct, st

#-------------------------------------------------------------------------
# End SFT
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# squick
#-------------------------------------------------------------------------

def squick(nptwo,fx,nf,ntap=None,kopt=None):

    """ 
    Sine multitaper routine. With a double length FFT constructs
    FT[sin(q*n)*x(n)] from F[x(n)], that is constructs the 
    FFT of the sine tapered signal. 
    
    The FFT should be performed previous to the call. 
    
    **Parameters**
    
    nptwo : float
        The twice signal length (2*npts)
    fx : ndarray, clomplex		
        The FFT of the signal (twice length)
    nf : int
        Number of frequency points for spec
    ntap : int, optional
        Constant number of tapers to average from
        if None, kopt is used.
        if > 0  Constant value to be used
        if <= 0 Use the kopt array instead
    ktop : ndarray, int [nf]
        array of integers, with the number of tapers
        at each frequency. 
   
    **Returns**
    
    spec : ndarray (nf,)	
        the spectral estimate
  
    **References**
    
    Based on the sine multitaper code of R. L. Parker.

    |

    """

    spec = np.zeros(nf,dtype=float)
    if (kopt is None and ntap is None):
        raise ValueError("Either kopt or ntap must exist")
    elif (kopt is None):
        if (ntap<1):
           ntap = int(3.0 + np.sqrt(float(nptwo/2))/5.0)
        kopt = np.ones(nf,dtype=int)*ntap

    #-------------------------------------------
    #  Loop over frequency
    #-------------------------------------------
    for m in range(nf):
      
        m2      = 2* (m)
        spec[m] = 0.
        klim    = kopt[m]
        ck      = 1./float(klim)**2

        #------------------------------------------------
        #  Average over tapers, parabolic weighting wk
        #------------------------------------------------
        for k0 in range(klim):
            k  = k0+1
            j1 = (m2+nptwo-k)%nptwo
            j2 = (m2+k)%nptwo

            zz = fx[j1] - fx[j2]
            wk = 1. - ck*float(k0)**2

            spec[m] = spec[m] + (np.real(zz)**2 + np.imag(zz)**2) * wk
        # end average tapers

        #-------------------------------------------------
        #  Exact normalization for parabolic factor
        #-------------------------------------------------

        spec[m] = spec[m] * (6.0*float(klim))/float(4*klim**2+3*klim-1)

    # end loop frequencies


    return spec, kopt

#-------------------------------------------------------------------------
# end squick
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
# squick2 - for cros spectra
#-------------------------------------------------------------------------

def squick2(nptwo,fx,nf,ntap=None,kopt=None):

    """ 
    Sine multitaper routine. With a double length FFT constructs
    FT[sin(q*n)*x(n)] from F[x(n)], that is constructs the 
    FFT of the sine tapered signal. 
    
    The FFT should be performed previous to the call. 

    **Parameters**
    
    nptwo : float
        The twice signal length (2*npts)
    fx : ndarray, complex [nptwo,2]		
        The FFT of the two signals (twice length)
    nf : int
        Number of frequency points for spec
    ntap : int, optional```
        Constant number of tapers to average from
        if > 0  Constant value to be used
        if None kopt used
        if <= 0 Use the kopt array instead
    kopt : ndarray, int [nf]
        array of integers, with the number of tapers
        at each frequency. 
   
    **Returns**
    
    spec : ndarray (nf,4)	
        the spectral estimates (first 2 columns)
        and the cross spectral estiamtes (last 2 columns)
  
    **References**
    
        Based on the sine multitaper code of R. L. Parker.

    |

    """

    sxy = np.zeros((nf,4),dtype=float)
    if (kopt is None and ntap is None):
        raise ValueError("Either kopt or ntap must exist")
    elif (kopt is None):
        if (ntap<1):
           ntap = int(3.0 + np.sqrt(float(nptwo/2))/5.0)
        kopt = np.ones(nf,dtype=int)*ntap

    #-------------------------------------------
    #  Loop over frequency
    #-------------------------------------------
    for m in range(nf):
      
        m2        = 2* (m)
        sxy[m,:]  = 0.
        klim      = kopt[m]
        ck        = 1./float(klim)**2

        #------------------------------------------------
        #  Average over tapers, parabolic weighting wk
        #------------------------------------------------

        for k0 in range(klim):
            k  = k0+1
            j1 = (m2+nptwo-k)%nptwo
            j2 = (m2+k)%nptwo

            z1 = fx[j1,0] - fx[j2,0]
            z2 = fx[j1,1] - fx[j2,1]
            wk = 1. - ck*float(k0)**2

            sxy[m,0] = sxy[m,0] + (np.real(z1)**2 + np.imag(z1)**2) * wk
            sxy[m,1] = sxy[m,1] + (np.real(z2)**2 + np.imag(z2)**2) * wk
            sxy[m,2] = sxy[m,2] + (np.real(z1)*np.real(z2) + np.imag(z1)*np.imag(z2)) * wk
            sxy[m,3] = sxy[m,3] + (np.real(z2)*np.imag(z1) - np.real(z1)*np.imag(z2)) * wk
 
        # end average tapers

        #-------------------------------------------------
        #  Exact normalization for parabolic factor
        #-------------------------------------------------

        sxy[m,:] = sxy[m,:] * (6.0*float(klim))/float(4*klim**2+3*klim-1)

    # end loop frequencies


    return sxy, kopt

#-------------------------------------------------------------------------
# end squick2
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# sadapt
#-------------------------------------------------------------------------

def sadapt(nptwo,fx,nf,df,initap,ntimes,fact):
    """
    Performs the (sine multitaper) adaptive spectral estimation
    From a basic pilot estimate, computes S" to be used
    in (13) of Riedel and Sidorenko (1995) for the 
    MSE spectrum.
  
    **Parameters**
    
    nptwo :	int
        The twice signal length (2*npts)
    fx : ndarray, complex [nptwo]		
        The FFT of the two signals (twice length)
    nf : int
        Number of frequency points for spec
    df : float
        Freq sampling
    initap : int
        Number of tapers to use for pilot estimate
  		Later we can add the spec result as test
    ntimes : int		
        number of iterations for estimate
    fact : float		
        degree of smoothing (def = 1.0)
   
    **Returns**

    spec : ndarray (nf)	
        the spectral estimate
    kopt : ndarray, int [nf]
        the number of tapers at each frequency. 
   
    **References**
    
    Based on the sine multitaper code of R. L. Parker.
  	
    **Calls**
    
    squick, north, curb

    |

    """

    #------------------------------------------------------
    # parabolic weighting
    #  c1, c2=(20*sqrt(1.2))**0.4 are constants 
    #  for parabolic weighting 
    #in subroutine quick
    #  for uniform weighting c1=1, c2=12.0**0.4=2.702
   
    c1=1.2000 
    c2=3.437

    #-------------------------------------
    # Get pilot estimate
    #-------------------------------------

    spec,kopt = squick(nptwo,fx,nf,initap)

    #------------------------------------------------------------
    #  Do the adaptive estimate. Find MSE iteratively.
    #  Estimate number of tapers at each freq for MSE spectrum
    #  Estimate 2nd derivative of S from ln S:
    #  To find |S"/S| use |theta"| + (theta')**2, theta=ln S
    #------------------------------------------------------------

    opt = np.zeros(nf,dtype=float)
    for iter in range(ntimes):

        y = np.log(spec)

        #-------------------------------------------------------------
        #  Estimate K, number of tapers at each freq for MSE spectrum
        #  R = S"/S -- use R = Y" + (Y')**2 , Y=ln S.
        #  Note  c2=3.437
        #-------------------------------------------------------------

        for j in range(nf):
      
            ispan  = int(kopt[j]*1.4)

            d1, d2 = north(nf,j-ispan, j+ispan, y)

            R    = (d2  + d1**2)/df**2
            ak   = float(kopt[j])/float(2*ispan)
            phi  = 720.0*ak**5*(1.0 - 1.286*ak + 0.476*ak**3 - 0.0909*ak**5)
            sigR = np.sqrt(phi/float(kopt[j])**5) / df**2

            opt[j] = c2/(df**4 *( R**2 + 1.4*sigR**2) /fact**2)** 0.2
        # end j loop

        #----------------------------------------------------------
        #  Curb runaway growth of Kopt near zeros of R
        #----------------------------------------------------------

        opt  = curb(nf,opt)
        kopt = np.maximum(opt,3.0)
        kopt = kopt.astype(int)
        #-----------------------------------------------------------
        #  Recompute spectrum with optimal variable taper numbers
        #-----------------------------------------------------------

        spec, kopt = squick(nptwo,fx,nf,kopt=kopt)

    # end iterations (ntimes)

    return spec, kopt
#-------------------------------------------------------------------------
# end sadapt
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
# sadapt2
#-------------------------------------------------------------------------

def sadapt2(nptwo,fx,nf,df,initap,ntimes,fact):
    """
    Performs the adaptive spectral estimation
    From a basic pilot estimate, computes S" to be used
    in (13) of Riedel and Sidorenko (1995) for the 
    MSE spectrum.

    **Parameters**
    
    nptwo :	int
        The twice signal length (2*npts)
    fx : ndarray, complex [nptwo,2]		
        The FFT of the two signals (twice length)
    nf : int
        Number of frequency points for spec
    df : float
        Freq sampling
    initap : int
        Number of tapers to use for pilot estimate
  		Later we can add the spec result as test
    ntimes : int		
        number of iterations for estimate
    fact : float		
        degree of smoothing (def = 1.0)
   
    **Returns**

    spec : ndarray (nf,4)	
        the spectral estimate  and coherence, phase
    kopt : ndarray, int [nf]
        the number of tapers at each frequency. 
   
    **References**
    
    Based on the sine multitaper code of R. L. Parker.
  	
    **Calls**
    
    squick, north, curb

    **Calls**
    
    squick2, orthog

    |

    """

    #------------------------------------------------------
    # parabolic weighting
    #  c1, c2=(20*sqrt(1.2))**0.4 are constants 
    #  for parabolic weighting 
    #in subroutine quick
    #  for uniform weighting c1=1, c2=12.0**0.4=2.702
   
    c1=1.2000 
    c2=3.437

    #-----------------------------------------------------------
    # Get pilot estimate
    #-----------------------------------------------------------

    spec, kopt = squick2(nptwo,fx,nf,initap)

    #------------------------------------------------------------
    #  Do the adaptive estimate. Find MSE iteratively.
    #  Estimate number of tapers at each freq for MSE spectrum
    #  Estimate 2nd derivative of S from ln S:
    #  To find |S"/S| use |theta"| + (theta')**2, theta=ln S
    #------------------------------------------------------------

    opt = np.zeros((nf,2),dtype=float)
    for iter in range(ntimes):
        for ipsd in range(2):

            y = np.log(spec[:,ipsd])

            #-------------------------------------------------------------
            #  Estimate K, number of tapers at each freq for MSE spectrum
            #  R = S"/S -- use R = Y" + (Y')**2 , Y=ln S.
            #  Note  c2=3.437
            #-------------------------------------------------------------

            for j in range(nf):
      
                ispan  = int(kopt[j]*1.4)

                d1, d2 = north(nf,j-ispan, j+ispan, y)

                R    = (d2  + d1**2)/df**2
                ak   = float(kopt[j])/float(2*ispan)
                phi  = 720.0*ak**5*(1.0 - 1.286*ak + 0.476*ak**3 - 0.0909*ak**5)
                sigR = np.sqrt(phi/float(kopt[j])**5) / df**2

                optj = c2/(df**4 *( R**2 + 1.4*sigR**2) /fact**2)** 0.2
 
                opt[j,ipsd] = optj
            # end j loop

            #----------------------------------------------------------
            #  Curb runaway growth of Kopt near zeros of R
            #----------------------------------------------------------

            opt2  = np.minimum(opt[:,0],opt[:,1])
            opt3  = curb(nf,opt2)
            kopt  = np.maximum(opt3,3.0)
            kopt  = kopt.astype(int)

            #-----------------------------------------------------------
            #  Recompute spectrum with optimal variable taper numbers
            #-----------------------------------------------------------

            spec, kopt = squick2(nptwo,fx,nf,kopt=kopt)

    # end iterations (ntimes)

    return spec, kopt
#-------------------------------------------------------------------------
# end sadapt
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# North
#-------------------------------------------------------------------------

def north(n, i1, i2, s):

    """
    Performs LS fit to s by
    a degree-two polynomial in an orthogonal basis.
    Function to be run with the Sine multitaper codes.

    **Returns**
    
    ds : float
        estimate of 1st derivative  ds/dn  at center of record
    dds : float
        estimate of 2nd derivative

    |

    """

    L     = i2 - i1 + 1
    el    = float(L)
    gamma = (el**2 - 1.0)/12.0
   
    u0sq = el
    u1sq = el*(el**2 - 1.0)/12.0
    u2sq = (el*(el**2 - 1.0)*(el**2- 4.0))/180.0
    amid= 0.5*(el + 1.0)
    dot0=0.0
    dot1=0.0
    dot2=0.0
    for kk in range(1,L+1):
       i = kk + i1 - 1 - 1

       # Negative or excessive index uses even function assumption

       if (i < 0):
           i = -i
       if (i > n-1):
           i  = 2*(n-1) - i
       dot0 = dot0 + s[i]
       dot1 = dot1 + (kk - amid) * s[i]
       dot2 = dot2 + ((kk - amid)**2 - gamma)*s[i]
       
    ds = dot1/u1sq
    dds = 2.0*dot2/u2sq

    return ds, dds
     
#-------------------------------------------------------------------------
# end North
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Curb
#-------------------------------------------------------------------------

def curb(n, v_in):

    """
    Takes n-long vector v[n] and rewrites it so that all points lie below
    the piece-wise linear function v(k) + abs(j-k), where v(k)
    is a local minimum in the original v.
    
    Effectively clips strong peaks and keeps slopes under 1 in
    magnitude.
    
    **Parameters**

    v_in : ndarray [n] 
        vector to be clipped, n-long

    **Returns**
    
    v : ndarray [n] 
        clipped vector

    |

    """

    n = np.shape(v_in)[0]
    v = np.copy(v_in)

    for j in range(1,n-1):

        # Scan series for local minimum
        if (v[j] < v[j+1] and v[j] < v[j-1]):
            vloc = v[j]

            #--------------------------------------------------------
            # this was done before, but for long n, it took too long
            # Revise series accordingly
            #     for k in range(n):
            #         v[k] = min(v[k], vloc+float(np.abs(j-k)))
            #--------------------------------------------------------
            
            # Revise series accordingly
            kmax = int(min(vloc,20))
            for k in range(-kmax,kmax+1):
                j2 = j+k
                if (j2>=0 and j2<n):
                   v[j2] = min(v[j2], vloc+float(np.abs(k)))
            # end k loop

        # end if minimum        
    # end j loop

    return v

#-------------------------------------------------------------------------
# end curb
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
# DATA_FILE - get data file from ZENODO repository
#-------------------------------------------------------------------------

def get_data(fname):
    """
    Utility function to download the data from the Zenodo repository
    with the direct URL path (fixed). 
    
    **Parameters**
    
    fname : char
        filename of the data to download
    
    **Returns**
    
    data : ndarray
        numpy array with the downloaded data
        In case of error, data = 0 is returned

    |

    """
    
    if (fname.find("v22")>-1):
        url = 'https://zenodo.org/record/6025794/files/v22_174_series.dat?download=1'
    elif (fname.find("hhe.dat")>-1):
        url = 'https://zenodo.org/record/6025794/files/sgc_vmm_hhe.dat?download=1'
    elif (fname.find("sgc_vmm.dat")>-1):
        url = 'https://zenodo.org/record/6025794/files/sgc_vmm.dat?download=1'
    elif (fname.find("sgc_surf")>-1):
        url = 'https://zenodo.org/record/6025794/files/sgc_surf.dat?download=1'
    elif (fname.find("sgc_mesetas")>-1):
        url = 'https://zenodo.org/record/6025794/files/sgc_mesetas.dat?download=1'
    elif (fname.find("PASC")>-1):
        url = 'https://zenodo.org/record/6025794/files/PASC.dat?download=1'
    elif (fname.find("_src")>-1):
        url = 'https://zenodo.org/record/6025794/files/mesetas_src.dat?download=1'
    elif (fname.find("crisanto")>-1):
        url = 'https://zenodo.org/record/6025794/files/crisanto_mesetas.dat?download=1'
    elif (fname.find("akima")>-1):
        url = 'https://zenodo.org/record/6025794/files/asc_akima.dat?download=1'
    elif (fname.find("ADO")>-1):
        url = 'https://zenodo.org/record/6025794/files/ADO.dat?download=1'
    else:
        data = -1
        
    data = np.loadtxt(url)
    
    return data

#-------------------------------------------------------------------------
# end DATA_FILE
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Examples - Copy example folder to user-defined folder
#-------------------------------------------------------------------------

def copy_examples(path="./multitaper-examples"):
    """
    Copy the examples folder, so the user can have access to the
    Notebooks and .py files
  
    Use `multitaper.utils.copy_examples()` function to copy all
    Notebooks and .py example files to local directory

    Install the examples for multitaper in the given location.

    WARNING: If the path exists, files will be overwritten. 
    Default path is `./multitaper-examples/` to avoid potential 
    overwrite of common folder names.
    Dependencies for the notebooks include
    - `matplotlib`
    - `scipy`
    - `numpy`
    These need to be available in the enviroment used. 

    **References**

       Codes based on an example from 
       Ben Mather, Robert Delhaye, within the PyCurious package. 

    |

    """
    import pkg_resources as pkg_res
    import os
    from distutils import dir_util

    ex_path = pkg_res.resource_filename(
        "multitaper", os.path.join("examples")
    )

    cex = dir_util.copy_tree(
         ex_path,
         path,
         preserve_mode=1,
         preserve_times=1,
         preserve_symlinks=1,
         update=0,
         verbose=0,
         dry_run=0,
    )

#-------------------------------------------------------------------------
# End copy examples folder
#-------------------------------------------------------------------------






