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




batch_size         = 128         
maxepoches         = 200    
weight_decay       = 1e-4
num_class           = 2
weight_decay       = 1e-4


learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
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


seed = 2020
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

############# Insert the specific data loading code here (cifar10,cifar100,SVHN or Catdog)

def depthwise_separable(x,params):
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
    
def build_model(datlist,classes=num_class):
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
        if tempSum>=arraySum*args.percentage:
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
def kernel_channel_wise_pca_conv(layerList,mode,r_channel=None):
    
    array_weight = layerList[0]
    array_bias = layerList[1]
    dim = array_weight.shape
    a = dim[0]
    b = dim[1]
    c = dim[2]
    d = dim[3]
    arrayold=array_weight.reshape([a*b*c,d])
    dataMat = np.vstack((arrayold,array_bias))
    if mode ==1:
        r_kernel = select_component(dataMat)
        X_p= PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)
        arraynew_weight = X_p[0:(a*b*c),:]
        arraynew_bias = X_p[-1,:]
        arraynew_weight1 = arraynew_weight.reshape([a,b,c,r_kernel])
        layerList[0] = arraynew_weight1
        layerList[1] = arraynew_bias
    if mode ==2:
        r_kernel = select_component(dataMat)
        X_p = PCA(n_components=r_kernel).fit(dataMat).transform(dataMat)
        arraynew_weight = X_p[:-1, :]
        arraynew_bias = X_p[-1, :]
        outDat_T = speical_PCA(arraynew_weight.T, a*b*r_channel)
        outDat = outDat_T.T
        arraynew_weight1 = outDat.real.reshape([a,b,r_channel,r_kernel])
        layerList[0] = arraynew_weight1
        layerList[1] = arraynew_bias
    return [layerList,r_kernel]    
    


####### load the pretain model of MobileNet on a specific dataset	
renew_model = load_model('/full_model/MobileNet_full.h5')

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

# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
renew_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# set callback
tb_cb = TensorBoard(histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]


ori_layer = ([32,64,128,128,256,256,512,512,512,512,512,512,1024,1024])
conv_index = conv_index_full[args.start:]

## PPCA process
for i in range(args.update):
    
    print("=" * 40 )
    print(f"start {i}-loop update")
    print("=" * 40 )
    
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
        if j in conv_index[1:]:
            [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 2,r_channel = tmp)
            pdtmp_conv.append(Conv_update)
            ptmp.append(p)
            tmp = p
    stati = ori_layer[:args.start]
    pca_update_layers = np.hstack((stati,ptmp))
    renew_model = build_model(pca_update_layers)
    renew_model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics = ['accuracy'])
    renew_model.summary()
    
    #for k in range(len(pd)):
    #    if k in conv_index:
    #        print(k)
    #        renew_model.layers[k].set_weights(pdtmp_conv[conv_index.index(k)])
    #    #if k in dense_index:
    #        #renew_model.layers[k].set_weights(pdtmp_dense[dense_index.index(k)])
    # 
    
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

