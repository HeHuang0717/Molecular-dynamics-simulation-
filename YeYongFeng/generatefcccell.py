# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:09:29 2019

@author: fz
"""

import numpy as np

def fcc_cell(a):
    b = 0.5*a
    coordinate = np.empty([14,3],dtype=np.float64)
    coordinate[0] = [b,b,b]
    coordinate[1] = [-b,b,b]
    coordinate[2] = [-b,-b,b]
    coordinate[3] = [b,-b,b]
    coordinate[4] = [b,b,-b]
    coordinate[5] = [-b,b,-b]
    coordinate[6] = [-b,-b,-b]
    coordinate[7] = [b,-b,-b]
    coordinate[8] = [b,0,0]
    coordinate[9] = [-b,0,0]
    coordinate[10] = [0,b,0]
    coordinate[11] = [0,-b,0]
    coordinate[12] = [0,0,b]
    coordinate[13] = [0,0,-b]
    return coordinate