import os
from setuptools import setup, find_packages
import subprocess
import logging

PACKAGE_NAME = 'aml'


setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    description='Additions to the Scikit-Learn Pipeline that is helpful for me, and maybe someone else!',
    author='Alexander Capstick',
    author_email='',
    long_description=open('README.txt').read(),
    install_requires=[
                        "numpy>=1.22",
                        "pandas>=1.4",
                        "matplotlib>=3.5",
                        "scikit-learn>=1.0",
                        "tqdm>=4.64",
                        "tensorboard>=2.9",
                        "pytorch-lightning>=1.6",
    ]
)