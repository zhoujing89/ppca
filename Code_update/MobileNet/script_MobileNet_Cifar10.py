import argparse
import os
import sys
import random
import numpy as np
#import pandas as pd
import tensorflow as tf
from PIL import Image

import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras import backend as K
from keras.layers import Input, Conv2D,GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.layers import DepthwiseConv2D,Conv2D, MaxPooling2D,Dropout,Flatten
from keras.models import load_model
from keras import optimizers,regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.utils import np_utils
from sklearn.decomposition import PCA
from scipy.io import loadmat as load


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--percentage", type=float, default=0.85, help="累计方差贡献率占比")
parser.add_argument("-u", "--update", type=int, default=1, help="更新次数")
parser.add_argument("-s", "--start", type=int, default=2, help="开始层数")
args = parser.parse_args()

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,allow_soft_placement=True)
session_conf.gpu_options.allow_growth=True                      #不全部占满显存, 按需分配 
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

date = '0729'

## 控制随机性
seed = 2020
np.random.seed(seed) # seed是一个固定的整数即可
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)

batch_size         = 128         
maxepoches         = 200    
weight_decay       = 1e-4


(X0,Y0),(X1,Y1) = cifar10.load_data()

from keras.utils import np_utils
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
   zca_whitening=False,  # apply ZCA whitening
   rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
   width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
   height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
   horizontal_flip=True,  # randomly flip images
   vertical_flip=False)  # randomly flip images
datagen.fit(X0)


def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0001

change_lr = LearningRateScheduler(scheduler)

sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)


## 定义深度可分类卷积
## 尝试对 depth-wise convolution 来一下 l2 regularization + weight decay
## s1代表步长；f2卷积核个数
def depthwise_separable(x,params):
    # f1/f2 filter size, s1 stride of conv
    (s1,f2) = params
    x = DepthwiseConv2D((3,3),strides=(s1[0],s1[0]), padding='same',
                        depthwise_initializer="he_normal",depthwise_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(f2[0]), (1,1), strides=(1,1), padding='same',
               kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x
    
def build_model(datlist,classes=10):
    """Instantiates the MobileNet.Network has two hyper-parameters
        which are the width of network (controlled by alpha)
        and input size.
        # Arguments
            alpha: optional parameter of the network to change the 
                width of model.
            shallow: optional parameter for making network smaller.
            classes: optional number of classes to classify images
                into.
    """
    IMSIZE = 32
    input_layer = Input([IMSIZE,IMSIZE,3])
    x = input_layer
    x = Conv2D(datlist[0],(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = depthwise_separable(x,params=[(1,),(datlist[1],)])
    x = depthwise_separable(x,params=[(1,),(datlist[2],)])
    x = depthwise_separable(x,params=[(1,),(datlist[3],)])
    x = depthwise_separable(x,params=[(1,),(datlist[4],)])
    x = depthwise_separable(x,params=[(1,),(datlist[5],)])
    x = depthwise_separable(x,params=[(2,),(datlist[6],)])
    
    x = depthwise_separable(x,params=[(1,),(datlist[7],)]) 
    x = depthwise_separable(x,params=[(1,),(datlist[8],)])
    x = depthwise_separable(x,params=[(1,),(datlist[9],)])
    x = depthwise_separable(x,params=[(1,),(datlist[10],)])
    x = depthwise_separable(x,params=[(1,),(datlist[11],)])        

            
    x = depthwise_separable(x,params=[(2,),(datlist[12],)])
    x = depthwise_separable(x,params=[(1,),(datlist[13],)])

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)
    output_layer=x
    model=Model(input_layer,output_layer)
    return model
    
## 根据累计方差贡献率输出给定主成分的个数
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
        
## 本文提出的方法对卷积层进行kernel wise和channel wise的pca降维
## 这个必须用新方法降维，因为会出现r_channel大于r_kernel的情况，此时python自带PCA函数无法解决
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
        r_kernel = select_component(dataMat)                        ## 根据累计方差贡献率对d降维
        X_p= PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)  ## 进行PCA降维，得到降维后新的矩阵，样本数*新的维度（主成分个数）
        arraynew_weight = X_p[0:(a*b*c),:]   ## 分离权重矩阵
        arraynew_bias = X_p[-1,:]   ## 分离偏置向量
        arraynew_weight1 = arraynew_weight.reshape([a,b,c,r_kernel]) ## 更新卷积层的权重，注意此时通道个数变为降维后的p1了，模型改变
        layerList[0] = arraynew_weight1    ## 更新后的权重参数
        layerList[1] = arraynew_bias       ## 更新后的偏置参数
    if mode ==2:
        r_kernel = select_component(dataMat)
        svd_1 = np.dot(dataMat,dataMat.T)           ## 该矩阵维数:a*b*c+1,a*b*c+1
        eigVals_1,eigVects_1=np.linalg.eig(svd_1)  #输出特征根和特征向量
        eigValInd_1=np.argsort(-eigVals_1)  #对特征跟从大到小进行排序,返回索引位置，和R里的order是一样的
        eigValInd_1=eigValInd_1[0:(a*b*r_channel+1)]   ## 此时进行的是channel wise的降维
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
        arraynew_weight1 = outDat[:-1].reshape([a,b,r_channel,r_kernel]) ## 更新卷积层的权重，注意此时通道个数变为降维后的p了，模型改变
  
        layerList[0] = arraynew_weight1    ## 更新后的权重参数
        layerList[1] = arraynew_bias       ## 更新后的偏置参数
    return [layerList,r_kernel]
    

renew_model = load_model('/home/pkustudent/notebooks/full_model/MobileNet_cifar10_full.h5')  #加载已经保存好的模型

print("=" * 40 )
print(f"finish load full model")
print("=" * 40 )

layer_name=[]
for layer in renew_model.layers:
    layer_name.append(layer.name)
conv_index_full = []
for i in range(len(layer_name)):
    tmp2 = layer_name[i].find("conv")
    if tmp2 ==0:
        conv_index_full.append(i)
print(conv_index_full)



## 开始进行pca降维，这里只有conv层，只对1*1的point wise conv进行降维，即上个block的conv_index
ori_layer = ([32,64,128,128,256,256,512,512,512,512,512,512,1024,1024])
conv_index = conv_index_full[args.start:]

for i in range(args.update):
    
    print("=" * 40 )
    print(f"start {i}-loop update")
    print("=" * 40 )
    
    current_model = renew_model
    pd = []  ## 初始化列表
    for layer in current_model.layers:
        weight = layer.get_weights()
        pd.append(weight)  ## 把每一层的weight append到列表里，有的layer里有参数，有的没有
    pdtmp_conv = []  ## 存储中间更新的权重列表
    ptmp = []   ## 存储中间层降的维度
    for j in range(len(pd)):
        if j == conv_index[0]:
            [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 1)
            pdtmp_conv.append(Conv_update)
            ptmp.append(p)
            tmp = p
        if j in conv_index[1:]:
            [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 2,r_channel = tmp)
            pdtmp_conv.append(Conv_update)
            ptmp.append(p)
            tmp = p
    stati = ori_layer[:args.start]
    pca_update_layers = np.hstack((stati,ptmp))      ## 更新后的模型新输入
    renew_model = build_model(pca_update_layers)  ## 新的模型结构
    
    #for k in range(len(pd)):
     #   if k in conv_index:
      #      print(k)
       #     renew_model.layers[k].set_weights(pdtmp_conv[conv_index.index(k)])
        #if k in dense_index:
            #renew_model.layers[k].set_weights(pdtmp_dense[dense_index.index(k)])
    
    renew_model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics = ['accuracy'])
    renew_model.summary()
        
    renew_model.fit(datagen.flow(X0, YY0, batch_size=batch_size),steps_per_epoch=X0.shape[0] // batch_size, epochs=1, 
                              validation_data=(X1,YY1),callbacks=[change_lr],
                              verbose =1, shuffle=False)

print("=" * 40 )
print(f"finish determine stucture, final train...")
print("=" * 40 )

filepath_1 = f'/home/pkustudent/notebooks/PPCA_update/PPCA_update/MobileNet/Cifar10/MobileNet-cifar10-percentage{args.percentage}-update{args.update}-start{args.start}-' + date

os.mkdir(filepath_1)

filepath = filepath_1 + '/weights-improvement-{epoch:02d}-{val_accuracy:.4f}.h5'

#filepath="/home/pkustudent/notebooks/PPCA_update/PPCA_update/AlexNet/Cifar10/PPCA_best_model_result/AlexNet-cifar10-percentage{args.percentage}-update{args.update}-start{args.start}-weights-improvement-{epoch:02d}-{val_accuracy:.4f}.h5"

checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
 
history = renew_model.fit(datagen.flow(X0, YY0, batch_size=batch_size),
                          steps_per_epoch=X0.shape[0] // batch_size,
                          epochs=maxepoches,
                          validation_data=(X1,YY1),
                          verbose =1,
                          shuffle=False,
                          callbacks=[checkpoint,change_lr])
