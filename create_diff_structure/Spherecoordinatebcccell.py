# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:58:45 2019

@author: fz
"""
import numpy as np
import math

def spherecoor_bcc_cell(a):
    """给定体心立方晶胞边长给出极坐标下的晶胞中9个原子坐标，体心为原点"""
    coordinate = np.zeros([9,3],np.float64)
    theta = math.acos(1./math.sqrt(3))
    phi = math.pi/4
    r = a*math.sqrt(3)*0.5
    """除第一个返回原点外，其余8个依次为1~8卦限对应的坐标"""
    coordinate[1] = [r,theta,phi]
    coordinate[2] = [r,theta,3*phi]
    coordinate[3] = [r,theta,5*phi]
    coordinate[4] = [r,theta,7*phi]
    coordinate[5] = [r,math.pi-theta,phi]
    coordinate[6] = [r,math.pi-theta,3*phi]
    coordinate[7] = [r,math.pi-theta,5*phi]
    coordinate[8] = [r,math.pi-theta,7*phi]
    return coordinate

def trans_bcc_cell(coordinate,theta,phi):
    """对原球坐标数据作旋转变化，绕z轴旋转phi角
    以及绕xoy平面内与坐标矢量垂直且过原点的直线为轴旋转theta角"""
    coordinate[1:9,1] += theta
    coordinate[1:9,2] += phi