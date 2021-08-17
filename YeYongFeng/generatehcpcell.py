# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:50:36 2019

@author: fz
"""

import numpy as np
import math

def hcp_cell(a):
    c = math.sqrt(8/3)*a*0.5
    coordinate = np.empty([17,3],dtype=np.float64)
    coordinate[0] = [0.5*a,-0.5*a/math.sqrt(3),0]
    coordinate[1] = [0,a/math.sqrt(3),0]
    coordinate[2] = [-0.5*a,-0.5*a/math.sqrt(3),0]
    coordinate[3] = [0,0,c]
    coordinate[4] = [a,0,c]
    trans_array = np.array([[0.5,-0.5*math.sqrt(3),0],[0.5*math.sqrt(3),0.5,0],[0,0,1]])
    for i in range(5,10):
        coordinate[i] = np.dot(trans_array,coordinate[i-1].T).T
        coordinate[i+6] = np.copy(coordinate[i])
    coordinate[10:17] = coordinate[3:10]
    coordinate[10:17,2] = -c
    return coordinate