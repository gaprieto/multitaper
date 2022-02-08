# multitaper - (before specpy)
Multitaper codes translated into Python. 

*multitaper* v.1.0.4

Germán A. Prieto

Departamento de Geociencias

Universidad Nacional de Colomnbia


A collection of modules for spectral analysis using the multitaper algorithm. 
The modules not only includes power spectral density (PSD) estimation with confidence intervals, but also multivariate problems including coherence, dual-frequency, correlations, and deconvolution estimation. Implementations of the sine and quadratic multitaper methods are also available. 

*multitaper* can also do:

**DPSS calculation** -  
    Calculates the discrete prolate functions (Slepian).

*Jacknife Errors** -  
    adaptively weighted jackknife 95% confidence intervals

**F-test** - 
    F-test of line components of the spectra

**Line reshape** - 
    reshapes the eigenft's around significant line components. 

**Dual-freq spectrum** - 
    Calculates the single trace dual freq spectrum (coherence and 
    phase). Dual frequency between two signals is also possible. 

**Coherence** - 
    Coherence between two signals. Ppssible conversion to time-domain. 

**Transfer function** - 
    Calculates the transfer function between two signals. 
    Possible conversion to time-domain. 

# Major updates
- v1.0.3 - Created data folder to run scripts and notebooks (in examples/) with correct path. 

# Installation
The *multitaper* package is composed of a number of Python modules. As of January 2022, multitaper can be installed using [conda](https://docs.conda.io/en/latest/). pip installation not yet available. You can also simply download the folder and add to your Python path. 

# With Conda:
```python
> conda install -c gprieto multitaper
```
# I recommend creating a virtual environment before:
```python
> conda create --name mtspec
> conda activate mtspec
> conda install -c gprieto multitaper
```

# With pip:
```python
> pip install multitaper
```


# Documentation 

A collection of Jupyter Notebooks is available to reproduce the figures
of the F90 paper (Prieto et al., 2009) and the Python version 
(Prieto 2022 under review). It has examples of a number of uses of 
the code as listed above. 
 
 
NOTES:
Uses Scipy for FFT (fftw). 
Uses Scipy for DPSS calculation. 

# Citation:
Please use this reference when citing the codes. 

Prieto, G.A. (2022). multitaper: A multitaper spectrum analysis package in Python. Seis. Res. Lett. Under review.

and/or

Prieto, G., Parker , R., & Vernon III, F. (2009). A Fortran 90 library for multitaper spectrum analysis. Computers & Geosciences, Vol. 35, pp. 1701–1710.
