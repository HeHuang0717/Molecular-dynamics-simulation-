from numpy import *
import matplotlib.pyplot as plt
from keras import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Flatten, AveragePooling2D, UpSampling2D, Conv2D,MaxPooling2D
from keras.layers.noise import GaussianDropout
from imageio import imread
import os
from sklearn.utils import shuffle


piclength = 100
picnumber = 2000

training_inputs = zeros([picnumber,piclength, piclength, 1])#事先测过 200张psi05， 113张psi95 图片格式40x40x3故需要3通道
training_results = zeros([picnumber, 2])
def vectorized_result(j): #矢量化描述概率， 便于神经网络得到一组概率值
    e = zeros(2) #概率
    e[j] = 1.0
    return e

samplestr = r"E:\huangmachinelearning\steptemp\Q8000debye200\N100\step\0.png"
i=0
for filename in os.listdir(samplestr):
    src = samplestr + "/" + filename
    # pic = imread(src)
    pic = imread(src).reshape( piclength, piclength, 1)
    pic -= pic.min()  # 观察数组，发现是0-255的数，所以得进行归一化，方便网络处理
    pic = (pic.astype(dtype='float')) / pic.max()
    training_inputs[i, :] = pic
    training_results[i, :] = vectorized_result(0)
    i = i+1# 向量化

samplestr = r"E:\huangmachinelearning\steptemp\Q8000debye200\N100\step\730.png"
for filename in os.listdir(samplestr):
    src = samplestr + "/" + filename
    pic = imread(src).reshape(piclength, piclength, 1)
    # pic
    pic -= pic.min()  # 观察数组，发现是0-255的数，所以得进行归一化，方便网络处理
    pic = (pic.astype(dtype='float')) / pic.max()
    training_inputs[i, :] = pic
    training_results[i, :] = vectorized_result(1)
    i = i+1# 向量化

training_inputs,training_results = shuffle(training_inputs,training_results, random_state=1337)


def init_net_conv_twolayer():
    global net
    net = Sequential()
    net.add(Conv2D(input_shape=(piclength,piclength,1), filters=16, kernel_size=[5, 5], activation='relu', padding='same'))#卷积层
    net.add(MaxPooling2D(pool_size=2))#平均池化
    net.add(Conv2D(filters=32, kernel_size=[5, 5], activation='relu', padding='same'))
    net.add(MaxPooling2D(pool_size=2))
    net.add(GaussianDropout(0.2))
    net.add(Flatten())
    net.add(Dense(800, activation='relu'))
    # net.add(Dense(50, activation='relu'))
    net.add(Dense(2, activation='softmax'))#sigmoid函数也尝试过
    net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])#binary_crossentropy也尝试过
    #SGD优化尝试下来效果不好，故报告里未给出
init_net_conv_twolayer()

net.summary()

history = net.fit(training_inputs, training_results, epochs=20, batch_size=100, validation_split=0.1)
#
f = open(r'E:\huangmachinelearning\steptemp\Q8000debye200\N100\temp.txt','w')
f.write(str(history.history))
f.close()

net.save(r'E:\huangmachinelearning\steptemp\Q8000debye200\N100\net.h5')#保存模型

#测试代码
from keras.models import load_model
import re
import pandas as pd
net_shc = load_model(r'E:\huangmachinelearning\steptemp\Q8000debye200\N100\net.h5')#调取模型
samplestr=r'E:\huangmachinelearning\steptemp\Q8000debye100\N100\step\test'
possibility = []
ifcrystal=[]
filen=[]
for filename in os.listdir(samplestr):
    src = samplestr + "/" + filename
    pic = imread(src).reshape(piclength,piclength, 1)
    jg =  pic
    jg -= jg.min()
    jg = (jg.astype(dtype='float')) / jg.max()
    jg = expand_dims(jg, axis=0)
    # 返回0/1的值 若光光predict的话只能得到0&1的概率值
    gailv = list(net_shc.predict(jg)[0])
    # jg_ = net_shc.predict_classes(jg)
    # jg_ = jg_[0]
    print( filename,net_shc.predict_classes(jg)[0], gailv )
    l = re.split("_|\.", filename)
    #文件名
    # filen.append(int(l[0]))
    filen.append(filename)
    #判断晶体
    ifcrystal.append( net_shc.predict_classes(jg)[0])
    #存储该路
    possibility.append(gailv)

# import numpy as np
# box = np.zeros((95,95))
# for i,filename in enumerate(filen):
#     if i !=0:
#         res = re.split('[_.]',filename)
#         box[int(res[1]),int(res[2])] =  ifcrystal[i]
# plt.imshow(box)
# plt.colorbar()
    # print(filename,i)


names = []
for i,filename in enumerate(filen):
    res = re.split('[_.]', filename)
    names.append(res[0])





# df = pd.DataFrame(possibility)
# df.join( pd.DataFrame(filen))
dfrate = pd.concat([pd.DataFrame(names ), pd.DataFrame(possibility) ], ignore_index=True, axis=1)
dfrate = pd.concat([pd.DataFrame(dfrate), pd.DataFrame(ifcrystal) ], ignore_index=True, axis=1)
dfrate.columns=['id','r1','r2','iscrystal']
dfrate.to_csv(r"E:\huangmachinelearning\steptemp\Q8000debye100\N100\result.csv",index =None)


dfrate = pd.read_csv(r"E:\huangmachinelearning\steptemp\Q8000debye100\N100\result.csv")

dfrate['id'] = dfrate['id'].astype('int')
dfrate.sort_values(by = 'id',inplace=True)
dfrate['groupnumber'] = (dfrate['id']/20).astype('int')


# dfrate = pd.read_csv(r"E:\huangmachinelearning\steptemp\Q4000debye400\N100\.csv")
fig = plt.figure()
mean = dfrate.groupby('groupnumber').mean().reset_index()
std = dfrate.groupby('groupnumber').std().reset_index()
plt.scatter(mean['id'], mean['r1'], label='N=100 crystal',marker="o",s =5 , edgecolors='r')
plt.scatter(mean['id'], mean['r2'], label='N=100 liquid',marker="o",s =5 , edgecolors='g')
plt.xlabel("temperature")
plt.ylabel("output possibility")
plt.legend()


# dfrate = dfrate.set_index('id').sort_index()
# dfrate.plot()
#


    # l=re.split("_|\.",filename)
    # i = i[0].split('_')  # 切开名称，得到xy的坐标，便于后续画图
    # x = int(l[1])  # 得到的是字符串格式的，所以得改为整型
    # y = int(l[2])
    # XLOC.append(x)
    # YLOC.append(y)
    # if jg == 0:
    #     plt.scatter(x, y, c='r', s=3)  # 非晶
    # else:
    #     plt.scatter(x, y, c='blue', s=3)  # 晶



# src =r"D:\Crystallization\machinelearning\test.png"
# pic = imread(src)
# plt.imshow(pic[:-30,:-30])
# plt.scatter(5*array(XLOC), 5*array(YLOC), c=5*array(ifcrystal)+1, marker='.', s=5,linewidths=1,alpha=0.1)


# plt.plot(ifcrystal)