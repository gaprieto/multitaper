Installation
============

The multitaper package is composed of a number of Python modules. As of January 2022, multitaper can be installed using conda. `pip` installation is also available. You can also simply download the folder and install and add to your Python path.

Dependencies
You will need Python 3.7+. The following packages are required:

`numpy`

`scipy`

Optional dependencies for plotting and example Notebooks:

`jupyter`

`matplotlib`

With Conda:

.. code-block:: console

   >> conda install -c gprieto multitaper

I recommend creating a virtual environment before:

.. code-block:: console

   >> conda create --name mtspec
   >> conda activate mtspec
   >> conda install -c gprieto multitaper

With pip:

.. code-block:: console

   >> pip install multitaper

Local install:
Download a copy of the codes from github

.. code-block:: console

    git clone https://github.com/gprieto/multitaper.git

or simply download ZIP file from https://github.com/gaprieto/multitaper and navigate to the directory and type

.. code-block:: console

   >> pip install .



