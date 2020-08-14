import argparse
import os
import sys

import numpy as np
import random
#import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from keras.datasets import cifar10, cifar100
from keras import backend as K
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler, TensorBoard,ModelCheckpoint

from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from pandas.core.frame import DataFrame
from PIL import Image
from sklearn.decomposition import PCA



from keras.layers import Activation,Conv2D, BatchNormalization, Dense
from keras.layers import Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
from keras import Model
from keras import optimizers, regularizers
import keras
from scipy.io import loadmat as load




parser = argparse.ArgumentParser()
parser.add_argument("-p", "--percentage", type=float, default=0.85, help="累计方差贡献率占比")
parser.add_argument("-u", "--update", type=int, default=3, help="更新次数")
parser.add_argument("-s", "--start", type=int, default=4, help="开始层数")
args = parser.parse_args()


session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,allow_soft_placement=True)
session_conf.gpu_options.allow_growth=True                      #不全部占满显存, 按需分配 
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


## 控制随机性
seed = 2020
np.random.seed(seed) # seed是一个固定的整数即可
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)

num_class = 10
weight_decay = 0.0005
n_batch = 128
maxepoches = 200
date = '0729'

(X0,Y0),(X1,Y1) = cifar10.load_data()

N0=X0.shape[0];N1=X1.shape[0]
X0 = X0.reshape(N0,32,32,3)/255
X1 = X1.reshape(N1,32,32,3)/255
YY0 = np_utils.to_categorical(Y0)
YY1 = np_utils.to_categorical(Y1)

## 数据标准化
mean = np.mean(X0,axis=(0,1,2,3))
std = np.std(X0,axis=(0,1,2,3))
X0 = (X0-mean)/(std+1e-7)
X1 = (X1-mean)/(std+1e-7)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    shear_range=0.5,
    zoom_range=0.2, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(X0)


def build_model(datlist):
    IMSIZE = 32
    input_shape = (IMSIZE, IMSIZE, 3)
    input_layer = Input(input_shape)
    x = input_layer
    #weight_decay = 0.0005

    x = Conv2D(datlist[0], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(datlist[1], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(datlist[2], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  
    
    x = Conv2D(datlist[3], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(datlist[4], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  
    
    x = Conv2D(datlist[5], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  
    
    x = Conv2D(datlist[6], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(datlist[7], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  
    
    x = Conv2D(datlist[8], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  
    
    x = Conv2D(datlist[9], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(datlist[10], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  
    
    x = Conv2D(datlist[11], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  
    
    x = Conv2D(datlist[12], [3, 3], padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(x)
    x = BatchNormalization()(x)   
    
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x) 
    x = Dense(datlist[13],kernel_regularizer=regularizers.l2(weight_decay),activation = 'relu')(x)
    x = BatchNormalization()(x)
    
    x = Dropout(0.5)(x)
    x = Dense(num_class)(x)
    x = Activation('softmax')(x)
    output_layer = x
    model_vgg16 = Model(input_layer, output_layer)
    return model_vgg16

learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20


def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
change_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)


#ori_layer = ([64,64,128,128,256,256,256,512,512,512,512,512,512,512])

#renew_model = build_model(ori_layer)
#renew_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# renew_model.summary()

# renew_model.fit_generator(datagen.flow(X0, YY0, batch_size=n_batch),
#                     steps_per_epoch=X0.shape[0] // n_batch, epochs=maxepoches, 
#                           validation_data=(X1,YY1),callbacks=[reduce_lr],verbose =1, shuffle=True)

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
        if tempSum>=arraySum*args.percentage: ##比较当前选取特征根的方差和与目标累计方差和
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



renew_model = load_model('/home/pkustudent/notebooks/full_model/VGG_cifar10_full.h5')  #加载已经保存好的模型


print("=" * 40 )
print("finish load full model")
print("=" * 40 )


## 提取最后一个maxpooling所在的index，目的是提取(第一个）全连接层进行降维时的输入r_channel
layer_name=[]
for layer in renew_model.layers:
    layer_name.append(layer.name)
conv_index = []
maxpooling_index = []
dense_index = []
for i in range(len(layer_name)):
    tmp1 = layer_name[i].find("max")
    tmp2 = layer_name[i].find("conv")
    tmp3 = layer_name[i].find("dense")
    if tmp1 ==0:
        maxpooling_index.append(i)
    if tmp2 ==0:
        conv_index.append(i)
    if tmp3 ==0:
        dense_index.append(i)
print(conv_index)
print(maxpooling_index)
print(dense_index)


conv_index = conv_index[args.start:]
dense_index = dense_index[0:1]
ori_layer = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
for i in range(args.update):
    
        
    print("=" * 40 )
    print(f"start {i}-loop update")
    print("=" * 40 )

    current_model = renew_model
    pd = []  ## 初始化列表
    for layer in current_model.layers:
        weight = layer.get_weights()
        pd.append(weight)  ## 把每一层的weight append到列表里，有的layer里有参数，有的没有
   
    #tmp = 128
    pdtmp = []  ## 存储中间更新的权重列表
    ptmp = []   ## 存储中间层降的维度
    
    
    for j in range(len(pd)): 
        if j == conv_index[0]:            ## 更新其他卷积层
            [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 1)
            pdtmp.append(Conv_update)
            ptmp.append(p)
            tmp = p
        if j in conv_index[1:]:
            [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 3,r_channel = tmp)
            pdtmp.append(Conv_update)
            ptmp.append(p)
            tmp = p
        
        if j in dense_index:
            ww = current_model.layers[maxpooling_index[-1]].output_shape
            tmpa = ww[1]
            #b = ww[2]
            [Dense_update,p] = kernel_channel_wise_pca_dense(pd[j],tmp,a = tmpa,mode = 1)
            pdtmp.append(Dense_update)
            ptmp.append(p)
            tmp = p
            
    stati = ori_layer[:args.start]
    pca_update_layers = np.hstack((stati,ptmp))      ## 更新后的模型新输入
    renew_model = build_model(pca_update_layers)  ## 新的模型结构
     
    #for k in range(len(pd)):
    #    if k in conv_index:
    #        renew_model.layers[k].set_weights(pdtmp[conv_index.index(k)])
    #    if k in dense_index:
    #        renew_model.layers[k].set_weights(pdtmp[-1])
    
    renew_model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics = ['accuracy'])
    renew_model.summary()
   
    renew_model.fit(datagen.flow(X0, YY0, batch_size=n_batch),steps_per_epoch=X0.shape[0] // n_batch, epochs=1, 
                              validation_data=(X1,YY1),callbacks=[change_lr],verbose =1, shuffle=False)

print("=" * 40 )
print(f"finish determine stucture, final train...")
print("=" * 40 )
    
filepath_1 = f'/home/pkustudent/notebooks/PPCA_update/PPCA_update/VGG/Cifar10/VGG-cifar10-percentage{args.percentage}-update{args.update}-start{args.start}-' + date

os.mkdir(filepath_1)

filepath = filepath_1 + '/weights-improvement-{epoch:02d}-{val_accuracy:.4f}.h5'

#filepath="/home/pkustudent/notebooks/PPCA_update/PPCA_update/AlexNet/Cifar10/PPCA_best_model_result/AlexNet-cifar10-percentage{args.percentage}-update{args.update}-start{args.start}-weights-improvement-{epoch:02d}-{val_accuracy:.4f}.h5"

checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
 
history = renew_model.fit(datagen.flow(X0, YY0, batch_size=n_batch),
                          steps_per_epoch=X0.shape[0] // n_batch,
                          epochs=maxepoches,
                          validation_data=(X1,YY1),
                          verbose =1,
                          shuffle=False,
                          callbacks=[checkpoint,change_lr])
