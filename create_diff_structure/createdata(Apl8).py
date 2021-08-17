from matplotlib.patches import Circle, PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d
from scipy import linalg
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
def Gaussian(x,y,mux,muy,sigma):
    """二元高斯函数暂作简化处理：1.sigmax=sigmay=sigma。2.ρ=0"""
    fun = 1./(2.*3.141592653589793*sigma**2)*np.exp(-((x-mux)**2+(y-muy)**2)/(2.*sigma**2))
    return fun

def make_3d_grid(x_space, y_space, z_space):
    "creates 3d_Grid in given xyz-space"
    return np.vstack(np.meshgrid(x_space, y_space, z_space)).reshape(3, -1).T

def fill_volume_bcc(x_limit, y_limit, z_limit, a):
    "fill given volume with BCC structure"
    calibration_factor = a * 2./np.sqrt(3)
    x_space = np.arange(0, 2*x_limit, 1.)
    y_space = np.arange(0, 2*y_limit, 1.)
    z_space = np.arange(0, 2*z_limit, 1.)
    first_grid = make_3d_grid(x_space, y_space, z_space)
    second_grid = np.copy(first_grid)
    second_grid += 1./2.
    crystal = np.vstack((first_grid, second_grid)) * calibration_factor
    condition = ((crystal[:, 0] <= x_limit)&
                 (crystal[:, 1] <= y_limit)&
                 (crystal[:, 2] <= z_limit))
    return crystal[condition]

def fill_volume_fcc(x_limit, y_limit, z_limit, a):
    "fill given volume with BCC structure"
    calibration_factor = a * 2./np.sqrt(2)
    x_space = np.arange(0, 2*x_limit, 1.)
    y_space = np.arange(0, 2*y_limit, 1.)
    z_space = np.arange(0, 2*z_limit, 1.)
    first_grid = make_3d_grid(x_space, y_space, z_space)
    second_grid = np.copy(first_grid)
    third_grid = np.copy(first_grid)
    fourth_grid = np.copy(first_grid)
    second_grid[:, 0:2] += 1./2.
    third_grid[:, 0] += 1./2.
    third_grid[:, 2] += 1./2.
    fourth_grid[:, 1:] += 1./2.
    crystal = np.vstack((first_grid,
                         second_grid,
                         third_grid,
                         fourth_grid)) * calibration_factor
    condition = ((crystal[:, 0] <= x_limit)&
                 (crystal[:, 1] <= y_limit)&
                 (crystal[:, 2] <= z_limit))
    return crystal[condition]
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

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def add_hcp_line(x_vec, y_coord, z_coord):
    "create atom line along x-axis with space 1"
    crystal_line = np.zeros((len(x_vec), 3))
    crystal_line[:, 0] = x_vec
    crystal_line[:, 1] = y_coord
    crystal_line[:, 2] = z_coord
    return crystal_line
def picknonzero(array_2d):
    """挑选稀疏矩阵array_2d中的非零元素行数列数保存到列表index中，
       i,j分别为元素在矩阵中的行数和列数"""
    index = []
    for i in range(array_2d.shape[0]):
        for j in range(array_2d.shape[1]):
            if array_2d[i,j] > 0.:
                index.append([i,j])
    return index
def add_hcp_layer(noa_x, noa_y, z_coord):
    "creates HCP Layer"
    x_vec = np.arange(0, int(round(noa_x)))
    crystal_volume = np.empty((0, 3))
    for y_coord in np.arange(0, noa_y, 2*np.sin(np.pi / 3.)):
        first_line = add_hcp_line(x_vec, y_coord, z_coord)
        second_line = add_hcp_line(x_vec + 1./2.,
                                   y_coord + np.sin(np.pi / 3.), z_coord)
        crystal_volume = np.vstack((crystal_volume, first_line))
        crystal_volume = np.vstack((crystal_volume, second_line))
    return crystal_volume

def fill_volume_hcp(x_space, y_space, z_space, a):
    "fill given volume with HCP structure"
    lattice_correct = np.sqrt(8/3)
    noa_x = int(round(x_space))
    noa_y = int(round(y_space / (np.sin(np.pi / 3.))))
    noa_z = int(round(z_space / (lattice_correct/2.)))
    crystal = np.empty((0, 3))
    unshifted_layer = True
    for z_coord in np.arange(0, noa_z+1, lattice_correct/2.):
        if unshifted_layer:
            cur_crystal = add_hcp_layer(noa_x + 1, noa_y + 1, z_coord)
            unshifted_layer = False
        else:
            cur_crystal = add_hcp_layer(noa_x + 1, noa_y + 1, z_coord)
            cur_crystal[:, 0] += 1./2.
            cur_crystal[:, 1] += 1./(2*np.sqrt(3))
            unshifted_layer = True
        crystal = np.vstack((crystal, cur_crystal * a))
        condition = ((crystal[:, 0] <= x_space)&
                     (crystal[:, 1] <= y_space)&
                     (crystal[:, 2] <= z_space))
    return crystal[condition]
def Rotation_matrix(loc,theta,phi,rho):    
    rx_theta=np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
    ry_phi=np.array([[np.cos(phi),0,-np.sin(phi)],[0,1,0],[np.sin(phi),0,np.cos(phi)]])
    rz_rho=np.array([[np.cos(rho),np.sin(rho),0],[-np.sin(rho),np.cos(rho),0],[0,0,1]])
    loc2=rx_theta.dot(ry_phi).dot(rz_rho).dot(loc)    
    return loc2
#window = signal.gaussian(51, std=7)
length=200
fd=fill_volume_bcc(length, length, length, 20)
#fig = plt.figure(figsize=(12,12))
#ax = fig.add_subplot(331, projection='3d') 
#ax.scatter(fd[:,0], fd[:,1], fd[:,2], c='r')
fd2=np.array(fd).T
loc_cart2=Rotation_matrix(fd2,0,0.0,0)
x_min=min(loc_cart2[0])
x_max=max(loc_cart2[0])
x_mid=(x_max+x_min)/2
y_min=min(loc_cart2[1])
y_max=max(loc_cart2[1])
y_mid=(y_max+y_min)/2
z_min=min(loc_cart2[2])
z_max=max(loc_cart2[2])
z_mid=(z_max+z_min)/2

X=loc_cart2[0]-x_mid
Y=loc_cart2[1]-y_mid
Z=loc_cart2[2]-z_mid



#ax.scatter(loc_cart2[0], loc_cart2[1], loc_cart2[2], c='b')
#
#p = Circle(( length/2, length/2),length, alpha=0.3)
#ax.add_patch(p)
#art3d.pathpatch_2d_to_3d(p, z=20, zdir="z")

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')

#ax1 = fig.add_subplot(332) 
a=100
x=np.zeros([a,a])
#limit=30
#
for i in range(len(X)):   
    if -50<X[i]<50 and -50<Y[i]<50:
        x[int(X[i]+50),int(Y[i]+50)]=gaussian(int(Z[i]),0, 30)   
#    
#    
#    
#x1=gaussianarray(x,1)
###for i in range(len(x1)):
###    for j in range(len(x1[i])):
###        x1[i][j]=x1[i][j]+0.1*random.random()
plt.imshow(x,"gray")



#ax1 = fig.add_subplot(333) 
#x=np.zeros([2*a,2*a])
#for i in range(len(fd2[0])):
#    x[int(a/2)+int(loc_cart2[0][i]),int(a/2)+int(loc_cart2[1][i])]=gaussian(int(loc_cart2[2][1]),20, 10)     
#x1=gaussianarray(x,1)
##for i in range(len(x1)):
##    for j in range(len(x1[i])):
##        x1[i][j]=x1[i][j]+0.1*random.random()
#plt.imshow(x1,"gray")
#
#plt.savefig("bcc(new).png",dpi=800,bbox_inches = 'tight')