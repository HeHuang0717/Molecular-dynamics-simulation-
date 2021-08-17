# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:48:58 2019

@author: Administrator
"""
from scipy import linalg
import random
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def Gaussian(x,y,mux,muy,sigma):
    """二元高斯函数暂作简化处理：1.sigmax=sigmay=sigma。2.ρ=0"""
    fun = 1./(2.*3.141592653589793*sigma**2)*np.exp(-((x-mux)**2+(y-muy)**2)/(2.*sigma**2))
    return fun

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

def bcc_cell(a):
    """给定体心立方晶胞边长给出极坐标下的晶胞中9个原子坐标，体心为原点"""
    coordinate = np.zeros([9,3],np.float64)
    """除第一个返回原点外，其余8个依次为1~8卦限对应的坐标"""
    coordinate[1] = [a,a,a]
    coordinate[2] = [a,-a,a]
    coordinate[3] = [-a,a,a]
    coordinate[4] = [-a,-a,a]
    coordinate[5] = [a,a,-a]
    coordinate[6] = [a,-a,-a]
    coordinate[7] = [-a,a,-a]
    coordinate[8] = [-a,-a,-a]
    return coordinate
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

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def Rotation_matrix(loc,theta,phi,rho):    
    rx_theta=np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
    ry_phi=np.array([[np.cos(phi),0,-np.sin(phi)],[0,1,0],[np.sin(phi),0,np.cos(phi)]])
    rz_rho=np.array([[np.cos(rho),np.sin(rho),0],[-np.sin(rho),np.cos(rho),0],[0,0,1]])
#    print(rx_theta,ry_phi,rz_rho,loc)
    loc2=rx_theta.dot(ry_phi).dot(rz_rho).dot(loc)    
#    *ry_phi*rz_rho
    return loc2


fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(341, projection='3d')    
#coordinate=spherecoor_bcc_cell(10)
#loc_cart=sph2cart(coordinate)[0,0,0],
#loc_cart=np.array([[2,1,1],[1,2,3]]).T
a=20 
coordinate=hcp_cell(a)
#coordinate=fcc_cell(a)
#coordinate=spherecoor_bcc_cell(a)
coordinate=np.array(coordinate).T
loc_cart2=Rotation_matrix(coordinate,1,1,1)
#for i in range(len(coordinate[0])):
#    r2=pow(loc_cart2[0][i],2)+pow(loc_cart2[1][i],2)+pow(loc_cart2[2][i],2)
#    print(r2)
ax.scatter(coordinate[0], coordinate[1], coordinate[2], c='r')
length=30
ax.set_zlim(-length, length)
ax.set_xlim(-length, length)
ax.set_ylim(-length, length)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax1 = fig.add_subplot(342, projection='3d') 
ax1.scatter(loc_cart2[0], loc_cart2[1], loc_cart2[2], c='b')


length=30
ax1.set_zlim(-length, length)
ax1.set_xlim(-length, length)
ax1.set_ylim(-length, length)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1 = fig.add_subplot(343) 

x=np.zeros([4*a,4*a])
for i in range(len(coordinate[0])):
    x[2*a+int(coordinate[0][i]),2*a+int(coordinate[1][i])]=20-int(coordinate[2][i])    
x1=gaussianarray(x,1)
for i in range(len(x1)):
    for j in range(len(x1[i])):
        x1[i][j]=x1[i][j]+0.1*random.random()
plt.imshow(x1,"gray")
plt.xticks([])
plt.yticks([])
ax1 = fig.add_subplot(344) 
#window = signal.gaussian(51, std=7)
x=np.zeros([4*a,4*a])
for i in range(len(coordinate[0])):
    x[2*a+int(loc_cart2[0][i]),2*a+int(loc_cart2[1][i])]=gaussian(int(loc_cart2[2][i]),20,20)    

x1=gaussianarray(x,1)
#for i in range(len(x1)):
#    for j in range(len(x1[i])):
#        x1[i][j]=x1[i][j]+0.1*random.random()
plt.imshow(x1,"gray")
plt.xticks([])
plt.yticks([])
plt.savefig("hcp(APL29).png",dpi=800,bbox_inches = 'tight')