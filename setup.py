# -*- coding: utf-8 -*-
"""
reference : https://github.com/pypa/sampleproject/blob/master/setup.py
"""

from setuptools import (setup, find_packages)
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='sflow',
      version='1.8.0.3',
      description='super slim and simple syntatic sugar for Tensorflow',
      long_description='텐서플로우 개취 탑재',

      # The project's main homepage.
      url='https://github.com/dade-ai/sflow',
      author='dade',
      author_email='aiplore@gmail.com',

      license='MIT',

      # The project's main homepage.

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3.6',
      ],

      packages=find_packages(),
      install_requires=[
          'snipy'
      ])
