#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib

# -------------------------------
""" True: Use CuPy 
    False:Not use CuPy """
use_cupy = False
# -------------------------------

def is_cupy():
    """ Returns: bool: defined this file value. """
    return use_cupy

def xp_factory():
    """ Returns: imported instance of cupy or numpy. """
    if is_cupy():
        return importlib.import_module('cupy')
    else:
        return importlib.import_module('numpy')

def report():
    """ report which is used cupy or numpy. """
    if is_cupy():
        print('import cupy !')
    else:
        print('import numpy !')
