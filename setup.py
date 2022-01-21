from setuptools import setup, find_packages

setup(version="1.0.0",
      name='multitaper',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'numpy',
          'scipy',
          'numba',
          'matplotlib',
          'jupyter',
          'ipykernel',
          'python>=3.7',
          'requests',
      ],
      author="German A. Prieto",
      author_email="gaprietogo@unal.edu.co",
      description="Multitaper codes translated into Python",
      license="GNU GENERAL PUBLIC LICENSE",
      url="https://github.com/gaprieto/multitaper",
      keywords=="spectral analysis, multitaper, coherence, deconvolution"
          "transfer function, line detection, jackknife",
      platforms='OS Independent',
      classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        ]
      )
