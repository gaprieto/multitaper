# `multitaper`
Multitaper codes translated into Python. 

`multitaper` v.1.1.2

Germán A. Prieto

Departamento de Geociencias, Universidad Nacional de Colomnbia


A collection of modules for spectral analysis using the multitaper algorithm. 
The modules not only includes power spectral density (PSD) estimation with confidence intervals, but also multivariate problems including coherence, dual-frequency, correlations, and deconvolution estimation. Implementations of the sine and quadratic multitaper methods are also available. 

*multitaper* can also do:

**DPSS calculation** - 
    Calculates the discrete prolate functions (Slepian).

**Jacknife Errors** - 
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
- v1.0.3 
   - Created data folder to run scripts and notebooks (in examples/) with correct path. 
- v1.0.8 
   - All modules, functions and classes are now documented with docstring.
   - Example Notebooks and .py files can now be installed
   - Data for examples is automatically downloaded from Zenodo repository.
- v1.1.0 
   - Complete (almost complete) documentation, via Sphinx.
   - All comments by reviewers addressed
   - To do: improve Python standard for loops (improve speed). 
- v1.1.1 
   - Replaced some for-loops with faster numpy code
   - More to be done.
- v1.1.2 
   - Improvements to setup.py, documentation and importing modules/Classes
   - Thanks to Pascal Audet for suggestions.

# Documentation
- Web page (https://multitaper.readthedocs.io) created using Sphinx and published via readthedocs.
- PDF can be found in the main folder (multitaper.pdf)

# Installation
The `multitaper` package is composed of a number of Python modules. As of January 2022, multitaper can be installed using [conda](https://docs.conda.io/en/latest/). **pip** installation is also available. You can also simply download the folder and install  and add to your Python path. 

### Dependencies

You will need **Python 3.8+**. The following packages are automatically installed:

- [`numpy`](http://numpy.org)
- [`scipy`](https://scipy.org)

__Optional dependencies__ for plotting and example Notebooks:

- [`jupyter`](https://jupyter.org/)
- [`matplotlib`](https://matplotlib.org/)

## I recommend creating a virtual environment before installing:
```python
> conda create --name mtspec python=3.8
> conda activate mtspec
```

## Install with Conda:
```python
> conda install -c gprieto multitaper
```

## Install with pip:
```python
> pip install multitaper
```

## Local install
Download a copy of the codes from github 
```
git clone https://github.com/gprieto/multitaper.git
```
or simply download ZIP file from https://github.com/gaprieto/multitaper
and navigate to the directory and type
```
pip install .
```

# Running the examples
A collection of Jupyter Notebooks and `.py` scripts are available 
to reproduce the figures of the F90 paper (Prieto et al., 2009) 
and the Python version  (Prieto 2022 under review). Data used in the 
examples is automatically downloaded from a Zenodo repository. 

To download the example folder, then python code
```
import multitaper.utils as utils
utils.copy_examples()
``` 
will create a folder `multitaper-examples/`. To run, install `matplotlib` and 
`jupyter` using `conda` and open the notebooks or run the python scripts 
(with the `multitaper` codes previously installed).
 
# Citation:
Please use this reference when citing the codes. 

Prieto, G.A. (2022). multitaper: A multitaper spectrum analysis package in Python. Seis. Res. Lett. Under review.

or

Prieto, G., Parker , R., & Vernon III, F. (2009). A Fortran 90 library for multitaper spectrum analysis. Computers & Geosciences, Vol. 35, pp. 1701–1710.
