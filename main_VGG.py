#!/usr/bin/env python
# coding: utf-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--percentage", type=float, default=0.85, help="variability")
parser.add_argument("-u", "--update", type=int, default=3, help="iteration")
parser.add_argument("-s", "--start", type=int, default=1, help="starting layer")
args = parser.parse_args()



percentage = args.percentage
update = args.update
start = args.start



seed = 2020
batch_size         = 128
maxepoches         = 200
weight_decay       = 0.0005
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
num_classes = 2
ori_layer = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]

# In[1]:


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
from keras.layers import add,Input, Conv2D,GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.layers import DepthwiseConv2D,Conv2D, MaxPooling2D,Dropout,Flatten
from keras.models import load_model
from keras import optimizers,regularizers
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.utils import np_utils
from sklearn.decomposition import PCA
from scipy.io import loadmat as load




np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)



def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
change_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

############# Insert the specific data loading code here (cifar10,cifar100,SVHN or Catdog)

def build_model(datlist):
    IMSIZE = 32
    input_shape = (IMSIZE, IMSIZE, 3)
    input_layer = Input(input_shape)
    x = input_layer

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
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    output_layer = x
    model_vgg16 = Model(input_layer, output_layer)
    return model_vgg16

def select_component(dataMat):
    meanVals=np.mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=np.cov(meanRemoved,rowvar=0)
    eigVals,eigVects=np.linalg.eig(covMat)
    sortArray=sorted(eigVals,reverse = True )
    arraySum=sum(sortArray)
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num



def speical_PCA(dataMat,r_channel):
    dataMat = dataMat - np.mean(dataMat, axis=0)
    svd_1 = np.dot(dataMat.T, dataMat)
    eigVals_1, eigVects_1 = np.linalg.eig(svd_1)
    eigVals_list_1 = []
    for i in range(len(eigVals_1)):
        eigVals_list_1.append([i, eigVals_1[i]])
    eigVals_list_1.sort(key=lambda x: x[1], reverse=True)
    maxeigVals_index_1 = []
    eigVals_r_channel = eigVals_list_1[0:r_channel]
    for i in eigVals_r_channel:
        maxeigVals_index_1.append(i[0])
    redEigVects_1 = eigVects_1[:, maxeigVals_index_1]
    output_Mat = np.dot(dataMat, redEigVects_1)
    return output_Mat

## function of PPCA for convolutional layer
def kernel_channel_wise_pca_conv(layerList, mode, r_channel=None):
    array_weight = layerList[0]
    array_bias = layerList[1]
    dim = array_weight.shape
    a = dim[0]
    b = dim[1]
    c = dim[2]
    d = dim[3]
    arrayold = array_weight.reshape([a * b * c, d])
    dataMat = np.vstack((arrayold, array_bias))

    if mode == 1:
        r_kernel = select_component(dataMat)
        X_p = PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)
        arraynew_weight = X_p[0:(a * b * c), :]
        arraynew_bias = X_p[-1, :]
        arraynew_weight1 = arraynew_weight.reshape([a, b, c, r_kernel])
    if mode == 2:
        r_kernel = r_channel
        X_p = PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)
        arraynew_weight = X_p[0:(a * b * c), :]
        arraynew_bias = X_p[-1, :]
        arrayold = arraynew_weight.reshape([a * b * r_channel, c])
        X_p = PCA(n_components=r_channel).fit(arrayold).transform(arrayold)
        arraynew_weight1 = X_p.reshape([a, b, r_channel, r_channel])
    if mode == 3:
        r_kernel = select_component(dataMat)
        X_p = PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)
        arraynew_weight = X_p[0:(a * b * c), :]
        arraynew_bias = X_p[-1, :]
        arrayold = arraynew_weight.reshape([a * b * r_kernel, c])
        X_p = speical_PCA(arrayold, r_channel)
        arraynew_weight1 = X_p.reshape([a, b, r_channel, r_kernel])
    layerList[0] = arraynew_weight1
    layerList[1] = arraynew_bias
    return [layerList, r_kernel]



## function of PPCA for fully connected layer
def kernel_channel_wise_pca_dense(layerList, r_reduce, mode, a=None):
    array_weight = layerList[0]
    array_bias = layerList[1]
    dataMat = np.vstack((array_weight, array_bias))
    r_kernel = select_component(dataMat)
    X_p = PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)
    arraynew_weight = X_p[:-1, :]
    arraynew_bias = X_p[-1, :]
    if mode == 1:
        r_channel = a * a * r_reduce + 1
    if mode == 2:
        r_channel = r_reduce+1
    outDat_T = speical_PCA(arraynew_weight.T,r_channel)
    outDat = outDat_T.T
    arraynew_weight1  = outDat.real
    layerList[0] = arraynew_weight1
    layerList[1] = arraynew_bias
    return [layerList, r_kernel]


####### load the pretain model of VGG on a specific dataset
renew_model = load_model('/full_model/VGG_full.h5') 

print("=" * 40 )
print("finish load full model")
print("=" * 40 )




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

## PPCA process
conv_index = conv_index[start:]
dense_index = dense_index[0:1]
for i in range(update):
    print("=" * 40 )
    print(f"start {i}-loop update")
    print("=" * 40 )
    current_model = renew_model
    pd = []
    for layer in current_model.layers:
        weight = layer.get_weights()
        pd.append(weight)

    pdtmp = []
    ptmp = []

    for j in range(len(pd)): 
        if j == conv_index[0]:
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
            
    stati = ori_layer[:start]
    pca_update_layers = np.hstack((stati,ptmp))
    renew_model = build_model(pca_update_layers)
     
    #for k in range(len(pd)):
    #    if k in conv_index:
    #        renew_model.layers[k].set_weights(pdtmp[conv_index.index(k)])
    #    if k in dense_index:
    #        renew_model.layers[k].set_weights(pdtmp[-1])
    
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
                    
filepath_1 = f'/percentage{percentage}-update{update}-start{start}'
if os.path.exists(f'/percentage{percentage}-update{update}-start{start}'):
    shutil.rmtree(f'/percentage{percentage}-update{update}-start{start}')
else:
    os.mkdir(f'/percentage{percentage}-update{update}-start{start}')

filepath = filepath_1 + '/weights-improvement-{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
 
history = renew_model.fit(train_generator, batch_size=batch_size,
                          epochs=maxepoches,
                          validation_data=validation_generator,
                          verbose =1,
                          shuffle=False,
                          callbacks=[checkpoint,change_lr])

