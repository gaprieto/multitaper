from setuptools import setup, find_packages

setup(
    name="multitaper",
    version="1.0.3",
    author="German A. Prieto",
    author_email="gaprietogo@unal.edu.co",
    description="Multitaper codes translated into Python",
    long_description_content_type="text/markdown",
    url="https://github.com/gaprieto/multitaper",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    data_files=[('data',['data/ADO.dat','data/PASC.dat','data/asc_akima.dat',
        'crisanto_mesetas.dat','d18O_1.dat','d18O_2.dat','mesetas_src.dat',
        'sgc_vmm_hhe.dat','v22_174_series.dat'])],
)   
