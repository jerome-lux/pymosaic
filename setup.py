# -*- coding=utf-8

from setuptools import setup, find_packages

setup(name='pymosaic',
      version='1.0',
      author='Jerome Lux',
      description='Module to create image mosaic',
      packages=find_packages(include=['pymosaic','pymosaic.*']),
      install_requires=['numpy>=1.18.5',
                        'PIL>=7.2.0',
                        'json>=2.0.9',
                        'numba>=0.50.1']
      )