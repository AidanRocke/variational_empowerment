#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 09:55:17 2018

@author: aidanrocke & ildefonsmagrans
"""

from distutils.core import setup

SRC_DIR = '.'

setup(
    name='vime_seed',
    version='0.5dev',
    package_dir={'vime_seed': SRC_DIR},
    packages=['vime_seed'],
    license='All rights reserved.',
    long_description=open('README.md').read(),
)