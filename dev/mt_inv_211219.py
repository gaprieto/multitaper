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

def cjk_fits(spec,nsamp,kspec,xi,C,Pk,iplot=0,df=1):
    """
    Performs the misfit calculation between observed Cjk and predicted Pjk
    """

    import numpy as np
    import scipy.interpolate as interp

    Pjk = cjk_pred(spec,nsamp,kspec,xi,Pk)

    nfft  = np.shape(Pjk)[1]

#    if (spec.ndim==1):
#        spec = spec[:,None]
#    L     = kspec*kspec
#    nfft  = np.shape(spec)[0]
#    fsamp = np.arange(-nsamp,nsamp+1)
#    Sw    = np.zeros((L,1),   dtype=float)

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
        ax2.plot(freq,spec)
#        plt.show()

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
    # Begin Least squares solution (QR factorization)
    #----------------------------------------------------

    for i in range(nfft):
       Cjk[:,0]    = C[:,i]
       #---------------------------------------------
       # Estimate positive spectrumm
       #---------------------------------------------
       cte_out  = optim.nnls(np.real(h1), 
                             np.real(Cjk[:,0]))[0]
       cte[i]  = np.real(cte_out) 

    #-----------------------------------------------------------
    # First find appropriate scaling of spectrum
    #    Newton's method
    #-----------------------------------------------------------

    spec  = np.copy(cte[:,None])
    #spec  = np.copy(qispec)

    x1   = 1
    f1   = cjk_fits(x1*spec,nsamp,kspec,xi2,C,Pk)
    print(x1,f1)
    x2   = 0.9
    f2   = cjk_fits(x2*spec,nsamp,kspec,xi2,C,Pk)
    print(x2,f2)
    df1  = (f1-f2)/(x1-x2)
    dx   = f1/df1
    while (dx>x1):
        dx = dx/2
    x0   = x1 - dx 
    print(df1,x0)

    for j in range(10):
        h      = 0.5*(x0-x1)
        while (abs(h)>x0):
            h = h/2
        x1     = x0+h
        x2     = x0-h
        f0     = cjk_fits(x0*spec,nsamp,kspec,xi2,C,Pk)
        f1     = cjk_fits(x1*spec,nsamp,kspec,xi2,C,Pk)
        f2     = cjk_fits(x2*spec,nsamp,kspec,xi2,C,Pk)
        df0_1  = (f0-f1)/(x0-x1)
        df0_2  = (f0-f2)/(x0-x2)
        df0    = (df0_1+df0_2)/2.0
        ddf0   = (f1-2*f0+f2)/(h**2)
        if (ddf0<1e-10):
            break
#        print('h', h)
#        print('df0 ', df0)
#        print('ddf0', ddf0)
        dx     = df0/ddf0
        while (dx>x0):
            dx = dx/2
        xnew   = x0-dx
        x2 = x1
        x1 = x0
        x0 = xnew
        f0 = cjk_fits(x0*spec,nsamp,kspec,xi2,C,Pk)
        print('xnew ,misfit', xnew, f0)

    print('Initial variance')
    err0 = cjk_fits(spec*0.0,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
    print('Zeros ', err0)

    print('Initial model')
    spec = spec * x0
    spec = spec[:,0]
    err = cjk_fits(spec,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
    print('var0 model ',err)

    print('Remove initial model')
    P0 = cjk_pred(spec*0.0,nsamp,kspec,xi2,Pk)
    C2  = C-P0
    err0 = cjk_fits(spec*0.0,nsamp,kspec,xi2,C2,Pk,iplot=1,df=df)
    print('model removed ',err)

    m0 = np.copy(spec)
    m1 = np.copy(spec)*0.0

#    fig = plt.figure()
#    ax0  = fig.add_subplot()
#
#    spec_save = np.copy(spec)
#    spec4 = np.copy(spec)
#    jac   = np.zeros(nfft)
#    for k0 in range(10):
#        for k in range(nfft):
#            spec2 = np.copy(spec)
#            spec2[k] = spec2[k]+0.05*spec2[k]
#            err2 = cjk_fits(spec2,nsamp,kspec,xi2,C,Pk)
#            spec3 = np.copy(spec)
#            spec3[k] = spec3[k]-0.05*spec3[k]
#            err3 = cjk_fits(spec3,nsamp,kspec,xi2,C,Pk)
#            derr2 = err2-err
#            derr3 = err3-err
#            emax = max(derr2,derr3)
#            if (emax<1e-8):
#                continue
#            if (derr2>derr3):
#                spec4[k] = spec3[k]
#            else:
#                spec4[k] = spec2[k]
#            jac[k] = (derr2)/(0.05*spec[k])
#            print(k, jac[k]) #derr2/(0.1*spec[k]), derr3/(0.1*spec[k]))
#        
#
#        spec = np.copy(spec4)
#        if (k0%5==0):
#            err = cjk_fits(spec,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
#            print('k0, err, var-red %3i %7.2f %5.1f' %(k0, err, np.abs(err-err0)/err0*100))
#        else:
#            err = cjk_fits(spec,nsamp,kspec,xi2,C,Pk,iplot=0)
#
#        ax0.plot(spec)

#    err = cjk_fits(spec,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
#    print('Manual', err)
#    spec = np.copy(spec_save)
#    test = optim.least_squares(cjk_fits, spec, args=(nsamp,kspec,xi2,C,Pk),
#                               loss='soft_l1',bounds=(0.0, np.inf),
#                               verbose=2)#,xtol=1e-3)
#    spec2 = test.x
#    err = cjk_fits(spec2,nsamp,kspec,xi2,C,Pk,iplot=1)

    print('Start method TNC')
#    bnd = scipy.optimize.Bounds(0, np.inf)
#    test = optim.minimize(cjk_fits, spec, args=(nsamp,kspec,xi2,C,Pk), 
#                     method='TNC', bounds=bnd,
#                     options={'maxiter': 100})#,'verbose':2})
#    spec2 = test.x
#    err = cjk_fits(spec2,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
#    print('TNC,err, var-red %3i %7.2f %5.1f' %(k0, err, np.abs(err-err0)/err0*100))

    print('Start method L-BFGS-B')
#    bnd = scipy.optimize.Bounds(0, np.inf)
#    test = optim.minimize(cjk_fits, spec, args=(nsamp,kspec,xi2,C,Pk), 
#                     method='L-BFGS-B', bounds=bnd,
#                     options={'maxiter': 100})#,'verbose':2})
#    spec2 = test.x
#    err = cjk_fits(spec2,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
#    print('BFGS,err, var-red %3i %7.2f %5.1f' %(k0, err, np.abs(err-err0)/err0*100))


    print('Start method trust-constr')
    bnd = scipy.optimize.Bounds(0, np.inf)
    test = optim.minimize(cjk_fits, m1, args=(nsamp,kspec,xi2,C2,Pk), 
                     method='trust-constr', bounds=bnd,
                     options={'maxiter': 100})#,'verbose':2})
    m2 = test.x
    spec2 = m0+m2
    err = cjk_fits(spec2,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
    print('Trust,err, var-red %3i %7.2f %5.1f' %(k0, err, np.abs(err-err0)/err0*100))


#    print('Start method Simulated Annealing')
#    bnd = np.zeros((len(spec),2))
#    bnd[:,1] = 1000 #np.inf
#    test = optim.dual_annealing(cjk_fits, x0=spec, bounds=bnd,args=(nsamp,kspec,xi2,C,Pk), 
#                     maxiter=100)
#    spec2 = test.x
#    err = cjk_fits(spec2,nsamp,kspec,xi2,C,Pk,iplot=1,df=df)
#    print('Ann,err, var-red %3i %7.2f %5.1f' %(k0, err, np.abs(err-err0)/err0*100))


    plt.show()
    return
    #-----------------------------------------------------------
    # Compare estimated Quadratic spectra and observed Cjk's
    #-----------------------------------------------------------

    nsamp = int(np.ceil(nw+1))

    s = 1
    s = 0.5
    err = cjk_fits(s*spec,nsamp,kspec,xi2,C,Pk)
    print(s,err)

    return
    fsamp = np.arange(-nsamp,nsamp+1)
    Pjk   = np.zeros((L,3),   dtype=complex)
    Cjk   = np.zeros((L,1),   dtype=complex)
    Sw    = np.zeros((L,1),   dtype=float)

    print(nsamp)
    print(fsamp)
    print(np.shape(spec))
    fig,(ax1,ax2) = plt.subplots(2,1)
    for i in range(nfft):
        i0 = (i-nsamp)
        i1 = (i+nsamp)
        i0 = fsamp+i
        if (i<nsamp):
            i0 = np.abs(i0)
        if (i>=nfft-nsamp):
            i0 = np.where(i0==nfft, 0, i0)
            i0 = np.where(i0>nfft, i0-2*(i0-nfft), i0)

        print(i,i0)

        Cjk[:,0] = C[:,i]

        Sw    = spec[i0,0]/3
        Sint  = interp.interp1d(fsamp, Sw, kind='quadratic') 
        Sw2   = Sint(xi*npts)
        Sw2   = Sw2[:,None]
        Pjk   = np.matmul(Pk,Sw2) 

        Sw    = qispec[i0,0]
        Sint  = interp.interp1d(fsamp, Sw, kind='quadratic') 
        Sw2   = Sint(xi*npts)
        Sw2   = Sw2[:,None]
        Pjk2  = np.matmul(Pk,Sw2) 

        m_cmp  = Cjk-Pjk
        m_real = np.sum(np.abs(np.real(m_cmp)))
        m_imag = np.sum(np.abs(np.imag(m_cmp)))

        m_cmp2  = Cjk-Pjk2
        m_real2 = np.sum(np.abs(np.real(m_cmp2)))
        m_imag2 = np.sum(np.abs(np.imag(m_cmp2)))

        print(i, m_real, m_imag, m_real2, m_imag2)

        #plt.plot(xi*npts+i,Sw2)
        #plt.plot(fsamp+i,Sw,'^')
        #plt.ylim((0,5))
        #plt.show()
        ax1.plot(i,m_real,'b.')
        ax1.plot(i,m_imag,'b^')

        ax1.plot(i,m_real2,'r.')
        ax1.plot(i,m_imag2,'r^')


    ax2.plot(spec)
    ax2.plot(qispec)

    plt.show()

    return

#-------------------------------------------------------------------------
# end qiinv
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
x = np.random.normal(0.0,1.0,npts)*0.1
t = np.arange(npts)*dt
y = 1*np.sin(2*np.pi*t*f0 )
#y = (1*np.sin(2*np.pi*t*f0 ) + 5.0*np.sin(2*np.pi*t*f1 )
#     + 1.0*np.cos(2*np.pi*t*f2) )

z = x+y

#----------------------------------------
# Create spectrum object
#----------------------------------------

xspec  = mtspec.mtspec(z,nw,kspec,dt)
qispec = xspec.qiinv()[0]

#----------------------------------------
# PSD inversion
#----------------------------------------

psd_inv(xspec.spec,qispec,xspec.freq,xspec.yk,xspec.wt,xspec.vn,xspec.lamb,xspec.nw)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(t,z)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(xspec.freq,xspec.spec)
ax.plot(xspec.freq,qispec)



plt.show()





