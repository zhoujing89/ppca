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
num_classes = 2
weight_decay = 1e-4
batch_size = 128
maxepoches = 200
ori_layer = ([96,256,384,384,256,4096,4096])


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



np.random.seed(seed) 
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)



def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0001

change_lr = LearningRateScheduler(scheduler)
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)


############# Insert the specific data loading code here (cifar10,cifar100,SVHN or Catdog)


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



# In[ ]:


####### load the pretain model of AlexNet on a specific dataset
renew_model = load_model('/full_model/AlexNet_full.h5')  

print("=" * 40 )
print("finish load full model")
print("=" * 40 )



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


## PPCA process
conv_index = conv_index_full[start:]
dense_index = dense_index_full[0:2]
for i in range(update):
    current_model = renew_model
    pd = []  
    for layer in current_model.layers:
        weight = layer.get_weights()
        pd.append(weight)  
    
    pdtmp_conv = []  
    ptmp = []   
        
    for j in range(len(pd)): 
        if j == conv_index[0]:          
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
    pca_update_layers = np.hstack((stati,ptmp))     
    renew_model = build_model(pca_update_layers)  
    
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

