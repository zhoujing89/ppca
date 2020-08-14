#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--percentage", type=float, default=0.85, help="累计方差贡献率占比")
parser.add_argument("-u", "--update", type=int, default=3, help="更新次数")
parser.add_argument("-s", "--start", type=int, default=1, help="开始层数")
args = parser.parse_args()


## 压缩率、pca更新次数、开始层数
percentage = args.percentage
update = args.update
start = args.start

## 其他可调参数
seed = 2020
num_classes = 2
weight_decay = 1e-4
batch_size = 128
maxepoches = 200
ori_layer = ([96,256,384,384,256,4096,4096])
#path = '/home/pkustudent/notebooks/tiny-imagenet-200'


import os
import sys
import shutil

import keras
import random
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler, TensorBoard,ModelCheckpoint
from keras.datasets import cifar10, cifar100
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, MaxPooling2D, add)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from pandas.core.frame import DataFrame
from PIL import Image
from sklearn.decomposition import PCA



## 控制随机性
np.random.seed(seed) # seed是一个固定的整数即可
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)


## 设置学习率、优化器
def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0001

change_lr = LearningRateScheduler(scheduler)
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)


IMSIZE=32

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    '/home/pkustudent/notebooks/CatDog/CatDog/validation',
    target_size=(IMSIZE, IMSIZE),
    batch_size=batch_size,
    class_mode='categorical')


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.5,
    rotation_range=30,
    zoom_range=0.2, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
    '/home/pkustudent/notebooks/CatDog/CatDog/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=batch_size,
    class_mode='categorical')

## 构建AlexNet模型
def build_model(datlist):
    IMSIZE = 32
    input_layer = Input([IMSIZE,IMSIZE,3])
    x = input_layer
    x = Conv2D(datlist[0],[3,3],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = MaxPooling2D([3,3], strides = [2,2])(x)    
    x = Conv2D(datlist[1],[3,3],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling2D([3,3], strides = [2,2])(x)
    x = Conv2D(datlist[2],[3,3],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = Conv2D(datlist[3],[3,3],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = Conv2D(datlist[4],[3,3],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = MaxPooling2D([3,3], strides = [2,2])(x)
    x = Flatten()(x)   
    x = Dense(datlist[5],activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(datlist[6],activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes,activation = 'softmax')(x) 
    output_layer=x
    model=Model(input_layer,output_layer)
    return model

# In[ ]:


### PPCA降维模块
def select_component(dataMat):
    meanVals=np.mean(dataMat,axis=0)  #，输入为n*p的矩阵，axis=0表示对每一列求平均值
    meanRemoved=dataMat-meanVals      #数据中心化
    covMat=np.cov(meanRemoved,rowvar=0)  #计算协方差矩阵，rowvar=0表示将列作为独立的变量
    eigVals,eigVects=np.linalg.eig(covMat)  #输出特征根和特征向量
    sortArray=sorted(eigVals,reverse = True )  #对特征跟从大到小进行排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i   ## 相加，返回左侧变量的值，这里就是tempSum的值
        num+=1
        if tempSum>=arraySum*percentage: ##比较当前选取特征根的方差和与目标累计方差和
            return num   ## 返回选取的特征根的个数，即主成分的个数


def kernel_channel_wise_pca_conv(layerList,mode,r_channel=None):
    
    array_weight = layerList[0]
    array_bias = layerList[1]
    dim = array_weight.shape                         ## 提取四个维度
    a = dim[0]                                ## 高度
    b = dim[1]                                ## 宽度
    c = dim[2]                                ## 深度
    d = dim[3]                                ## 卷积核个数
    arrayold=array_weight.reshape([a*b*c,d])         ## 变成二维，按照d进行pca
    dataMat = np.vstack((arrayold,array_bias))  ## 将权重矩阵和偏置向量进行拼接，拼接后的维度：（a*b*c+1）*d
   
    if mode ==1:
        r_kernel = select_component(dataMat)                        ## 根据累计方差贡献率计算要提取的主成分个数
        X_p= PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)  ## 进行PCA降维，得到降维后新的矩阵，样本数*新的维度（主成分个数）
        arraynew_weight = X_p[0:(a*b*c),:]   ## 分离权重矩阵
        arraynew_bias = X_p[-1,:]   ## 分离偏置向量
        arraynew_weight1 = arraynew_weight.reshape([a,b,c,r_kernel]) ## 更新卷积层的权重，注意此时通道个数变为降维后的p1了，模型改变
        layerList[0] = arraynew_weight1    ## 更新后的权重参数
        layerList[1] = arraynew_bias       ## 更新后的偏置参数
    if mode ==2:
        r_kernel = r_channel
        X_p= PCA(n_components=r_channel).fit(dataMat).transform(dataMat)  ## 进行PCA降维，得到降维后新的矩阵，样本数*新的维度（主成分个数）
        arraynew_weight = X_p[0:(a*b*c),:]   ## 分离权重矩阵
        arraynew_bias = X_p[-1,:]   ## 分离偏置向量
        arrayold=arraynew_weight.reshape([a*b*r_channel,c])         ## 变成二维，按照c进行pca，这里的维度是c，要补齐到p2，是为了和偏置合并时对齐
        X_p= PCA(n_components=r_channel).fit(arrayold).transform(arrayold)  ## 按照p进行PCA降维，得到降维后新的矩阵，样本数*新的维度（主成分个数）
        arraynew_weight1 = X_p.reshape([a,b,r_channel,r_channel]) ## 更新卷积层的权重，注意此时通道个数变为降维后的p了，模型改变
        layerList[0] = arraynew_weight1    ## 更新后的权重参数
        layerList[1] = arraynew_bias       ## 更新后的偏置参数
    if mode ==3:
        r_kernel = select_component(dataMat)                        ## 根据累计方差贡献率计算要提取的主成分个数
        X_p= PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)  ## 进行PCA降维，得到降维后新的矩阵，样本数*新的维度（主成分个数）
        arraynew_weight = X_p[0:(a*b*c),:]   ## 分离权重矩阵
        arraynew_bias = X_p[-1,:]   ## 分离偏置向量
        arrayold=arraynew_weight.reshape([a*b*r_kernel,c])         ## 变成二维，按照c进行pca，这里的维度是c，要补齐到p2，是为了和偏置合并时对齐
        X_p= PCA(n_components=r_channel).fit(arrayold).transform(arrayold)  ## 按照p进行PCA降维，得到降维后新的矩阵，样本数*新的维度（主成分个数）
        arraynew_weight1 = X_p.reshape([a,b,r_channel,r_kernel]) ## 更新卷积层的权重，注意此时通道个数变为降维后的p了，模型改变
        layerList[0] = arraynew_weight1    ## 更新后的权重参数
        layerList[1] = arraynew_bias       ## 更新后的偏置参数
    return [layerList,r_kernel]


## 新方法下的PCA，X’X和XX’进行特征分解
def kernel_channel_wise_pca_dense(layerList,r_reduce,mode,a = None):
    
    array_weight = layerList[0]
    array_bias = layerList[1]
    dataMat = np.vstack((array_weight,array_bias))  ## 样本数*120，这个样本数是最后一个maxpooling之后的维度
    
    svd_1 = np.dot(dataMat,dataMat.T)           ## 该矩阵维数:a*b*c+1,a*b*c+1
    eigVals_1,eigVects_1=np.linalg.eig(svd_1)  #输出特征根和特征向量
    eigValInd_1=np.argsort(-eigVals_1)  #对特征跟从大到小进行排序,返回索引位置，和R里的order是一样的
    r_kernel = select_component(dataMat)                        ## 根据累计方差贡献率对d降维
    if mode ==1:  ## 如果是第一个全连接层
       
        r_channel = a*a*r_reduce+1   ## 这里的a、b就是最后一个maxpooling层的kernel大小
        eigValInd_1=eigValInd_1[0:r_channel]   ## 此时进行的是channel wise的降维
        redEigVects_1=eigVects_1[:,eigValInd_1]   #返回排序后特征值对应的特征向量redEigVects（主成分）

    if mode ==2:  ## 如果是第二个全连接层及以后
        r_channel = r_reduce+1
        eigValInd_1=eigValInd_1[0:r_channel]   ## 此时进行的是channel wise的降维
        redEigVects_1=eigVects_1[:,eigValInd_1]   #返回排序后特征值对应的特征向量redEigVects（主成分）
   
    
    svd_2 = np.dot(dataMat.T,dataMat)           #矩阵维数：d*d
    eigVals_2,eigVects_2=np.linalg.eig(svd_2)  #输出特征根和特征向量
    eigValInd_2=np.argsort(-eigVals_2)  #对特征跟从大到小进行排序,返回索引位置，和R里的order是一样的
    eigValInd_2=eigValInd_2[0:r_kernel]   ## 根据kernel wise降维结果选择
    redEigVects_2=eigVects_2[:,eigValInd_2]   #返回排序后特征值对应的特征向量redEigVects（主成分）
    
    tmp = np.dot(redEigVects_1.T,dataMat)
    outDat = np.dot(tmp,redEigVects_2)
    outDat = outDat.real  ## 只返回实部  维数：a*b*r_channel+1,r_kernel
    arraynew_bias = outDat[-1,:]   ## 分离偏置向量
    arraynew_weight1 = outDat[:-1] ## 更新卷积层的权重，注意此时通道个数变为降维后的p了，模型改变
  
    layerList[0] = arraynew_weight1    ## 更新后的权重参数
    layerList[1] = arraynew_bias       ## 更新后的偏置参数
    return [layerList,r_kernel]


# In[ ]:


renew_model = load_model('/home/pkustudent/notebooks/full_model2/AlexNet_CatDog_full.h5')  #加载已经保存好的模型

print("=" * 40 )
print("finish load full model")
print("=" * 40 )



## 查看卷积层、池化层和全连接层的index
layer_name=[]
for layer in renew_model.layers:
    layer_name.append(layer.name)
conv_index_full = []
maxpooling_index_full = []
dense_index_full = []
for i in range(len(layer_name)):
    tmp1 = layer_name[i].find("max")
    tmp2 = layer_name[i].find("conv")
    tmp3 = layer_name[i].find("dense")
    if tmp1 ==0:
        maxpooling_index_full.append(i)
    if tmp2 ==0:
        conv_index_full.append(i)
    if tmp3 ==0:
        dense_index_full.append(i)
print(conv_index_full)
print(maxpooling_index_full)
print(dense_index_full)


## 进行PPCA降维
conv_index = conv_index_full[start:]
dense_index = dense_index_full[0:2]
for i in range(update):
    current_model = renew_model
    pd = []  ## 初始化列表
    for layer in current_model.layers:
        weight = layer.get_weights()
        pd.append(weight)  ## 把每一层的weight append到列表里，有的layer里有参数，有的没有
    
    pdtmp_conv = []  ## 存储中间更新的权重列表
    pdtmp_dense = []  ## 存储中间更新的权重列表
    ptmp = []   ## 存储中间层降的维度
        
    for j in range(len(pd)): 
        if j == conv_index[0]:            ## 更新其他卷积层
            [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 1)
            pdtmp_conv.append(Conv_update)
            ptmp.append(p)
            tmp = p
            print(j,p)
            #print(Conv_update)
        if j in conv_index[1:]:
            [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 3,r_channel = tmp)
            pdtmp_conv.append(Conv_update)
            ptmp.append(p)
            tmp = p
            print(j,p)
            #print(Conv_update)
        
        if j == dense_index[0]:
            ww = current_model.layers[maxpooling_index_full[2]].output_shape
            tmpa = ww[1]
            #b = ww[2]
            [Dense_update,p] = kernel_channel_wise_pca_dense(pd[j],tmp,a = tmpa,mode = 1)
            pdtmp_dense.append(Dense_update)
            ptmp.append(p)
            tmp = p
            print(j,p)
            #print(Conv_update)
            
        if j == dense_index[1]:
            [Dense_update,p] = kernel_channel_wise_pca_dense(pd[j],tmp,mode = 2)
            pdtmp_dense.append(Dense_update)
            ptmp.append(p)
            tmp = p
            print(j,p)
            #print(Conv_update)
    stati = ori_layer[:start]
    pca_update_layers = np.hstack((stati,ptmp))      ## 更新后的模型新输入
    renew_model = build_model(pca_update_layers)  ## 新的模型结构
    
    #for k in range(len(pd)):
    #    if k in conv_index:
    #        renew_model.layers[k].set_weights(pdtmp_conv[conv_index.index(k)])
    #    if k in dense_index:
    #        renew_model.layers[k].set_weights(pdtmp_dense[dense_index.index(k)])
    renew_model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics = ['accuracy'])
    renew_model.summary()
    
    renew_model.fit(train_generator, batch_size=batch_size,
                          epochs=1,
                          validation_data=validation_generator,
                          verbose =1,
                          shuffle=False,
                          callbacks=[change_lr])


print("=" * 40 )
print(f"finish determine stucture, final train...")
print("=" * 40 )


# In[ ]:


filepath_1 = f'/home/pkustudent/notebooks/PPCA_update/PPCA_update/AlexNet/CatDog/AlexNet-CarDog-percentage{percentage}-update{update}-start{start}-new'
if os.path.exists(f'/home/pkustudent/notebooks/PPCA_update/PPCA_update/AlexNet/CatDog/AlexNet-CarDog-percentage{percentage}-update{update}-start{start}-new'):
    shutil.rmtree(f'/home/pkustudent/notebooks/PPCA_update/PPCA_update/AlexNet/CatDog/AlexNet-CarDog-percentage{percentage}-update{update}-start{start}-new')
else:
    os.mkdir(f'/home/pkustudent/notebooks/PPCA_update/PPCA_update/AlexNet/CatDog/AlexNet-CarDog-percentage{percentage}-update{update}-start{start}-new')

filepath = filepath_1 + '/weights-improvement-{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
 
history = renew_model.fit(train_generator, batch_size=batch_size,
                          epochs=maxepoches,
                          validation_data=validation_generator,
                          verbose =1,
                          shuffle=False,
                          callbacks=[checkpoint,change_lr])

