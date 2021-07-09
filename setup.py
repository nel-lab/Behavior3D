#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:49:38 2021

@author: jimmytabet
"""

from setuptools import setup, find_packages

setup(
      name='Behavior3D',
      version='1.0',
      packages=find_packages(include=['Behavior3D', 'Behavior3D.*']),
      install_requires=['']
      )