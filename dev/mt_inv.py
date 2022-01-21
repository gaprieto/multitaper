#-------------------------------------------------------------------------
# qiinv
#-------------------------------------------------------------------------

def cjk_pred(spec,nsamp,kspec,xi,Pk):
    """
    Performs the misfit calculation between observed Cjk and predicted Pjk
    """

    import numpy as np
    import scipy.interpolate as interp

    if (spec.ndim==1):
        spec = spec[:,None]
    L     = kspec*kspec
    nfft  = np.shape(spec)[0]
    fsamp = np.arange(-nsamp,nsamp+1)
    Pjk   = np.zeros((L,nfft),dtype=complex)
    Sw    = np.zeros((L,1),   dtype=float)

    for i in range(nfft):
        i0 = (i-nsamp)
        i1 = (i+nsamp)
        i0 = fsamp+i
        if (i<nsamp):
            i0 = np.abs(i0)
        if (i>=nfft-nsamp):
            i0 = np.where(i0==nfft, 0, i0)
            i0 = np.where(i0>nfft, i0-2*(i0-nfft), i0)

        Sw       = spec[i0,0]
        Sint     = interp.interp1d(fsamp, Sw, kind='quadratic') 
        Sw2      = Sint(xi)
        Sw2      = Sw2[:,None]
        P0       = np.matmul(Pk,Sw2)
        Pjk[:,i] = P0[:,0] 

    return Pjk 

def model_L1(spec):
    """
    Performs the misfit calculation between observed Cjk and predicted Pjk
    """

    import numpy as np
    import scipy.interpolate as interp

    mnorm = np.sum(np.abs(spec))

    return mnorm

def misfit_L1(spec,nsamp,kspec,xi,C,Pk):
    """
    Performs the misfit calculation between observed Cjk and predicted Pjk
    """

    import numpy as np
    import scipy.interpolate as interp

    Pjk = cjk_pred(spec,nsamp,kspec,xi,Pk)

    nfft  = np.shape(Pjk)[1]

    m_real = np.zeros(nfft)
    m_imag = np.zeros(nfft)

    for i in range(nfft):

        m_cmp     = C[:,i]-Pjk[:,i]
        m_real[i] = np.sum(np.abs(np.real(m_cmp)))
        m_imag[i] = np.sum(np.abs(np.imag(m_cmp)))

    misfit = np.sum(m_real) + np.sum(m_imag)

    return misfit

def misfit_L2(spec,nsamp,kspec,xi,C,Pk):
    """
    Performs the misfit calculation between observed Cjk and predicted Pjk
    """

    import numpy as np
    import scipy.interpolate as interp

    Pjk = cjk_pred(spec,nsamp,kspec,xi,Pk)

    nfft  = np.shape(Pjk)[1]

    m_real = np.zeros(nfft)
    m_imag = np.zeros(nfft)

    for i in range(nfft):

        m_cmp     = C[:,i]-Pjk[:,i]
        m_real[i] = np.sum(np.abs(np.real(m_cmp)))
        m_imag[i] = np.sum(np.abs(np.imag(m_cmp)))

    misfit = np.sqrt(np.sum(m_real**2) + np.sum(m_imag**2))

    return misfit

def cjk_scale(cte, s0,nsamp,kspec,xi,C,Pk):
    """
    Performs the misfit calculation between observed Cjk and predicted Pjk
    """

    import numpy as np
    import scipy.interpolate as interp

    spec = s0*cte
    Pjk = cjk_pred(spec,nsamp,kspec,xi,Pk)

    nfft  = np.shape(Pjk)[1]

    m_real = np.zeros(nfft)
    m_imag = np.zeros(nfft)

    for i in range(nfft):

        m_cmp     = C[:,i]-Pjk[:,i]
        m_real[i] = np.sum(np.abs(np.real(m_cmp)))
        m_imag[i] = np.sum(np.abs(np.imag(m_cmp)))

    misfit = np.sum(m_real) + np.sum(m_imag)

    return misfit

def cs_penalty(spec,nsamp,kspec,xi,C,Pk,alpha):
    """
    Performs the Compressed sensing (CS) penalty function 
    calculation following
    P = ||Pjk*spec -Cjk||2 + alpha*||spec||1 
    """

    import numpy as np

    misfit = misfit_L2(spec,nsamp,kspec,xi,C,Pk)
    mnorm  = model_L1(spec)

    Pfun = misfit + alpha*mnorm

    return Pfun 


def cjk_fits(spec,nsamp,kspec,xi,C,Pk,iplot=0,df=1):
    """
    Performs the misfit calculation between observed Cjk and predicted Pjk
    """

    import numpy as np
    import scipy.interpolate as interp

    Pjk = cjk_pred(spec,nsamp,kspec,xi,Pk)

    nfft  = np.shape(Pjk)[1]

    m_real = np.zeros(nfft)
    m_imag = np.zeros(nfft)
    freq   = np.arange(nfft)*df

    for i in range(nfft):

        m_cmp     = C[:,i]-Pjk[:,i]
        m_real[i] = np.sum(np.abs(np.real(m_cmp)))
        m_imag[i] = np.sum(np.abs(np.imag(m_cmp)))

    if (iplot==1):
        fig,(ax1,ax2) = plt.subplots(2,1)
        ax1.plot(freq,m_real,'b.')
        ax1.plot(freq,m_imag,'r.')
        ax2.plot(freq,spec,'-.')

    misfit = np.sum(m_real) + np.sum(m_imag)

    return misfit

def psd_inv(spec,qispec,freq,yk,wt,vn,lamb,nw):

    """
    First try on PSD inversion procedure

    INPUT
        spec[nfft,0]    the adaptive spectrum (so far)
        yk[npts,kspec]  multitaper eigencoefficients, complex
  	wt[nf,kspec]	the weights of the different coefficients. 
  			input is the original multitaper weights, 
  			from the Thomson adaptive weighting. 
  	vn[npts,kspec]  the slepian sequences
        lambda(kspec]   the eigenvalues of the Slepian sequences
        nw              The time-bandwisth product
  	
    OUTPUT
  	qispec[nfft,0]	the QI spectrum
  	ds[nfft,0]	the estimate of the first derivative
  	dds[nfft,0]	the estimate of the second derivative
  
    FIRST WRITTEN
  	German Prieto
  	December 13, 2021
    """

    import scipy.optimize as optim
    import scipy
    import scipy.interpolate as interp


    npts  = np.shape(vn)[0] 
    kspec = np.shape(vn)[1]
    nfft  = np.shape(yk)[0]
    nsc   = nfft/npts
    nsamp = int(np.ceil(nw*nsc+1))
    df    = freq[2]-freq[1]

    print('nfft, npts, nsamp ', nfft, npts,nsamp)
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
    xi2  = xi*nfft
    dxi  = xi[2]-xi[1]
    f_qi = scipy.fft.fftfreq(nfft)

    print(np.shape(f_qi),np.shape(spec))
    for k in range(kspec):
        xk[:,k] = wt[:,k]*yk[:,k];
        for i in range(nxi):
            om = 2.0*np.pi*xi[i]
            ct,st = utils.sft(vn[:,k],om) 
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

            Pk[m,:] = np.conjugate(Vj[:,k]) * (Vj[:,i]);

    Pk[:,0]         = 0.5 * Pk[:,0];
    Pk[:,nxi-1]     = 0.5 * Pk[:,nxi-1];

    #-----------------------------------------------------------
    # First 
    #     Get estimate of constant spectrum
    #-----------------------------------------------------------

    hk     = np.zeros((L,3),   dtype=complex)
    hcte   = np.ones((nxi,1),  dtype=float)
    Cjk    = np.zeros((L,1),   dtype=complex)
    cte    = np.zeros(nfft)
    h1     = np.matmul(Pk,hcte) #* dxi

    #----------------------------------------------------
    # Begin QI least squares solution (QR factorization)
    #----------------------------------------------------

    for i in range(nfft):
       Cjk[:,0]    = C[:,i]
       #---------------------------------------------
       # Estimate positive spectrumm
       #---------------------------------------------
       cte_out = optim.nnls(np.real(h1), 
                             np.real(Cjk[:,0]))[0]
       cte[i]  = np.real(cte_out) 

    #-----------------------------------------------------------
    # First find appropriate scaling of spectrum
    #    Newton's method
    #-----------------------------------------------------------

    #spec  = np.copy(cte[:,None])
    spec  = np.copy(qispec)

    print('Initial variance')
    err0 = cjk_fits(spec*0.0,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
    print('Zeros ', err0)

    print('Scale PSD - minimize_scalar')
    bnd       = optim.Bounds(0, np.inf)
    test      = optim.minimize_scalar(cjk_scale, bounds=bnd, 
                    args=(spec,nsamp,kspec,xi2,C,Pk), method='brent') 
    psd_scale = test.x
    m0        = spec[:,0]*psd_scale
    err       = cjk_fits(m0,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
    print('Optimal scale factor ', psd_scale)
    print('Scale Trust,err, var-red %7.2f %5.1f'
             %(err, (err-err0)/err0*100))


    print('Remove initial model')
    P0   = cjk_pred(m0,nsamp,kspec,xi2,Pk)
    C2   = C-P0
    err0 = cjk_fits(spec*0.0,nsamp,kspec,xi2,C2,Pk,iplot=1,df=df)
    print('model removed ',err)
    m1 = np.copy(m0)*0.0

    #-----------------------------------------------
    # Potential methods for nonlinear optimization
    #-----------------------------------------------
    # Algorithm - optim.minimize
    # Method TNC
    # Method L-BFGS-B
    # Method trust-constr
    # Method optim.dual_annealing

    # Fit with the entire model
    print(np.shape(m0))
    print('Start method trust-constr')
    bnd = scipy.optimize.Bounds(0, np.inf)
    test = optim.minimize(cjk_fits, m0, args=(nsamp,kspec,xi2,C,Pk), 
                     method='trust-constr', bounds=bnd,
                     options={'maxiter': 100})#,'verbose':2})
    m2    = test.x
    err   = cjk_fits(m2,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
    print('Trust,err, var-red %7.2f %5.1f' %(err, (err-err0)/err0*100))


    # Fit with the initial model removed
    print('Model removed trust-constr')
    bnd  = optim.Bounds(-m0, np.inf)
    test = optim.minimize(cjk_fits, m1, args=(nsamp,kspec,xi2,C2,Pk), 
                     method='trust-constr',bounds=bnd,
                     options={'maxiter': 100})
    m3    = test.x
    spec2 = m0+m3
    err   = cjk_fits(spec2,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
    print('Model removed Trust, var-red %7.2f %5.1f' %(err, (err-err0)/err0*100))

    fig2 = plt.figure()
    axf1 = fig2.add_subplot(211)
    axf2 = fig2.add_subplot(212)
    # Now, do the Compressed Sensing approach
    for alpha in [20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20 ]:
        print('CS trust-constr, alpha',alpha)
        bnd   = optim.Bounds(-m0, np.inf)
        test  = optim.minimize(cs_penalty, m1, 
                     args=(nsamp,kspec,xi2,C2,Pk,alpha), 
                     method='trust-constr',bounds=bnd,
                     options={'maxiter': 100})
        m4    = test.x
        spec2 = m0+m4
        misfit = misfit_L2(spec2,nsamp,kspec,xi2,C,Pk)
        mnorm  = model_L1(m4)

        axf1.semilogx(alpha,misfit,'ko')
        axf2.semilogx(alpha,mnorm,'ko')

        print('alpha, L2-misfit, mnorm ', alpha, misfit, mnorm)
        err   = cjk_fits(spec2,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
        print('CS Trust, var-red %7.2f %5.1f' %(err, (err-err0)/err0*100))

    plt.show()
    return

#-------------------------------------------------------------------------
# end psd_inv
#-------------------------------------------------------------------------



"""
Initial test code for actual inversion of the spectrum using 
the quadratic approach. 
Note: This is in testing

December 2021
Germán A. Prieto

"""


import multitaper.mtspec as mtspec
import multitaper.utils as utils
import multitaper.mtcross as mtcross
import numpy as np
import matplotlib.pyplot as plt

npts  = 101
f0    = 0.20
f1    = 0.23  #0.24
f2    = 0.30
dt    = 1.0
kspec = 6
nw    = 4.0
W     = nw/npts
ph   = np.random.normal(0.0,np.pi,1)

print(W)

#----------------------------------------
# Create fake data
#----------------------------------------
x = np.random.normal(0.0,1.0,npts)*1.0
t = np.arange(npts)*dt
#y = 1*np.sin(2*np.pi*t*f0 )
y = (1*np.sin(2*np.pi*t*f0 ) + 2.0*np.sin(2*np.pi*t*f1 )
     + 1.0*np.cos(2*np.pi*t*f2) )

z = x+y

#----------------------------------------
# Create spectrum object
#----------------------------------------

xspec  = mtspec.mtspec(z,nw,kspec,dt)
qispec = xspec.qiinv()[0]

#----------------------------------------
# PSD inversion
#----------------------------------------

psd_inv(xspec.spec,qispec,xspec.freq,xspec.yk,
        xspec.wt,xspec.vn,xspec.lamb,xspec.nw)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(t,z)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(xspec.freq,xspec.spec)
ax.plot(xspec.freq,qispec)



plt.show()





