Installation
============

The multitaper package is composed of a number of Python modules. As of January 2022, multitaper can be installed using conda. `pip` installation is also available. You can also simply download the folder and install and add to your Python path.

Dependencies
You will need Python 3.8+. The following packages are automatically installed:

`numpy`

`scipy`

Optional dependencies for plotting and example Notebooks:

`jupyter`

`matplotlib`

I recommend creating a virtual environment before installing:

.. code-block:: console

   >> conda create --name mtspec python=3.8
   >> conda activate mtspec

Install with Conda:

.. code-block:: console

   >> conda install -c gprieto multitaper

Install with pip:

.. code-block:: console

   >> pip install multitaper

Local install:
Download a copy of the codes from github

.. code-block:: console

    git clone https://github.com/gprieto/multitaper.git

or simply download ZIP file from https://github.com/gaprieto/multitaper and navigate to the directory and type

.. code-block:: console

   >> pip install .



