import cv2
import numpy as np
from plotting import plotList
import time
import os
import ctypes
from ctypes import cdll
from numpy.ctypeslib import ndpointer

lib = cdll.LoadLibrary('./libLBP.so')
lib.solve.restype = ndpointer(dtype=ctypes.c_double, shape=(256,))
path = "hellow, world"
ress = lib.solve(path,len(path))
print(ress)
