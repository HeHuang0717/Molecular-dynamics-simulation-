# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:54:19 2019

@author: fz
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import random

def Gaussian(x,y,mux,muy,sigma):
    """二元高斯函数暂作简化处理：1.sigmax=sigmay=sigma。2.ρ=0"""
    fun = 1./(2.*3.141592653589793*sigma**2)*np.exp(-((x-mux)**2+(y-muy)**2)/(2.*sigma**2))
    return fun

def picknonzero(array_2d):
    """挑选稀疏矩阵array_2d中的非零元素行数列数保存到列表index中，
       i,j分别为元素在矩阵中的行数和列数"""
    index = []
    for i in range(array_2d.shape[0]):
        for j in range(array_2d.shape[1]):
            if array_2d[i,j] > 0.:
                index.append([i,j])
    return index

def gaussianarray(array_2d,sigma):
    """稀疏矩阵高斯化，将每个非零元素作为二元高斯函数的峰值，得到一个新的矩阵
       设矩阵array_2d中有l个非零元素，每个点对应一个只有期望(mux,muy)不同的高斯函数乘以权重
       设非零元素为(H1,H2,H3,...,Hl),各函数分别为f1,f2,f3,...,fl
       可以认为每个非零元素是各函数在该点对应函数值的加和，即Hi=∑ki*fi(xi,yi),i=1,2,...l
       ki为权重，xi,yi直接用Hi对应的行、列数代入，解线性方程组得到权重{k}，
       则新矩阵的各元素就是各高斯函数的线性叠加"""
    index = picknonzero(array_2d)
    length = len(index)
    num_of_peak = np.arange(0,len(index), dtype = np.int16)
    """每个峰值可以认为是该点对应高斯函数的峰值加上其他峰值对应高斯函数在该点的函数值，
       以下是求解线性方程组Ak=B的过程"""
    A = np.empty(shape = [length,length], dtype = np.float64)
    B = np.empty(shape = length, dtype = np.float64)
    for i in num_of_peak:
        for j in num_of_peak:
            A[i,j] = Gaussian(index[i][0],index[i][1],index[j][0],index[j][1],sigma)
        B[i] = array_2d[index[i][0],index[i][1]]
    k = linalg.solve(A,B)
    gaussian_array = np.copy(array_2d)
    for i in range(gaussian_array.shape[0]):
        for j in range(gaussian_array.shape[1]):
            if gaussian_array[i,j] == 0.:
                for n in num_of_peak:
                    """新矩阵的每个元素表示为各峰值对应的高斯函数在该位置的函数值的线性组合"""
                    gaussian_array[i,j] += k[n]*Gaussian(i,j,index[n][0],index[n][1],sigma)
    return gaussian_array
center=50
radius=30
def hexagon(center,radius):
    d=radius*0.5
    h=radius*0.8660254
    c=[center,center]
    c1=[center-int(radius),center]
    c2=[center-round(d),center+round(h)]
    c3=[center+round(d),center+round(h)]
    c4=[center+int(radius),center]
    c5=[center+round(d),center-round(h)]
    c6=[center-round(d),center-round(h)]    
    return c,c1,c2,c3,c4,c5,c6
#    for i in range(7):
#        location=[]
        
c,c1,c2,c3,c4,c5,c6 =hexagon(center,radius)     

x=np.zeros([100,100])
x[c[0],c[1]]=3
x[c1[0],c1[1]]=3
x[c2[0],c2[1]]=2
x[c3[0],c3[1]]=3
x[c4[0],c4[1]]=0.5
x[c5[0],c5[1]]=3
x[c6[0],c6[1]]=3
x1=gaussianarray(x,1)
for i in range(len(x1)):
    for j in range(len(x1[i])):
        x1[i][j]=x1[i][j]+0.1*random.random()
plt.subplot(211)
plt.imshow(x,"gray")
plt.subplot(212)
plt.imshow(x1,"gray")