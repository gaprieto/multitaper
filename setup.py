from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multitaper",
    version="1.0.2",
    author="German A. Prieto",
    author_email="gaprietogo@unal.edu.co",
    description="Multitaper codes translated into Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gaprieto/multitaper",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
)
