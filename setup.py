#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import with_statement

import sys

if sys.version_info < (2 , 5):
    sys.exit('Python 2.5 or greater is required.')

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


setup(name='RLToolbox' ,
      version="2.0.7" ,
      description='rl toolbox' ,
      long_description="RL toolbox",
      author='wuyupei' ,
      author_email='840302039@qq.com' ,
      maintainer='wuyupei' ,
      maintainer_email='840302039@qq.com' ,
      url='https://github.com/jjkke88/RL_toolbox' ,
      packages=find_packages(exclude=["tests.*", "tests"]),
      license="GPL 3.0" ,
      platforms=['any'] ,
      classifiers=[]
      )
