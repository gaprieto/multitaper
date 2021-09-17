def reshape(self,fcrit=0.95,p=None):   #p_in,yk_in,vn,fcrit=0.95):
    """
    reshape the yk's based on the F-test of line compenents
    """

    if (p is None):
        print('Doing F test')
        p        = self.ftest()[1]
    yk       = np.copy(self.yk)
    vn       = self.vn
    npts     = self.npts
    kspec    = self.kspec
    nfft     = self.nfft
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

    sk = np.abs(yk)**2

    #-----------------------------------------------------------------
    # Calculate adaptive spectrum
    #-----------------------------------------------------------------

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


