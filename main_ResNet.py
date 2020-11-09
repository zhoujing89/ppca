#!/usr/bin/env python
# coding: utf-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--percentage", type=float, default=0.85, help="variability")
parser.add_argument("-u", "--update", type=int, default=3, help="iteration")
parser.add_argument("-s", "--start", type=int, default=1, help="starting layer")
parser.add_argument("-m", "--blocklayer", type=int, default=1, help="starting block")
args = parser.parse_args()

percentage = args.percentage
update = args.update
start = args.start
blocklayer = args.blocklayer


seed = 2020
stack_n            = 5
layers             = 6 * stack_n + 2
num_class           = 2
batch_size         = 128
maxepoches             = 200
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4
ori_layer = [16,16,16,32,32,32,32,32,64,64,64,64,64]
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



def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
change_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

############# Insert the specific data loading code here (cifar10,cifar100,SVHN or Catdog)

def build_model(datlist):
    IMSIZE = 32
    input_layer = Input([IMSIZE,IMSIZE,3])
    x = input_layer
    x = Conv2D(datlist[0], (3, 3), strides=(1,1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
               kernel_initializer="he_normal")(x)
    for _ in range(5):
        b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(datlist[1],kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(datlist[2],kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
        kernel_initializer="he_normal")(a1)
        
        x = add([x,conv_2])

    b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    a0 = Activation('relu')(b0)
    conv_1 = Conv2D(datlist[3],kernel_size=(3,3),strides=(2,2),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
    b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
    a1 = Activation('relu')(b1)
    conv_2 = Conv2D(datlist[4],kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a1)
    
    projection = Conv2D(datlist[5],kernel_size=(1,1),strides=(2,2),padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
    x = add([projection,conv_2])

    for _ in range(1,5):
        b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(datlist[6],kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(datlist[7],kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a1)
        x = add([x,conv_2])

    b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    a0 = Activation('relu')(b0)
    conv_1 = Conv2D(datlist[8],kernel_size=(3,3),strides=(2,2),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
    b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
    a1 = Activation('relu')(b1)
    conv_2 = Conv2D(datlist[9],kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a1)
    
    projection = Conv2D(datlist[10],kernel_size=(1,1),strides=(2,2),padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
    x = add([projection,conv_2])    

    for _ in range(1,5):
        b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(datlist[11],kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(datlist[12],kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a1)
        x = add([x,conv_2])

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(num_class,activation='softmax',kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
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


####### load the pretain model of ResNet on a specific dataset
renew_model = load_model('/full_model/ResNet_full.h5') 

print("=" * 40 )
print("finish load full model")
print("=" * 40 )

layer_name=[]
for layer in renew_model.layers:
    layer_name.append(layer.name)

def output_layer(layer_name):
    conv_index = []
    if args.blocklayer ==1:
        for i in range(len(layer_name)):
            tmp2 = layer_name[i].find("conv")
            if tmp2 ==0:
                conv_index.append(i)
        list1 = [conv_index[11],conv_index[22]]
        list2 = conv_index[13:22]+conv_index[24:33]
    if args.blocklayer ==2:
        for i in range(len(layer_name)):
            tmp2 = layer_name[i].find("conv")
            if tmp2 ==0:
                conv_index.append(i)
        list1 = [conv_index[22]]
        list2 = conv_index[24:33]
    return(list1,list2)

[list1,list2] = output_layer(layer_name)

ori_layer = [16,16,16,32,32,32,32,32,64,64,64,64,64]
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

    pdtmp_list1 = []
    pdtmp_list2 = []
    ptnew = ori_layer[:args.start]
    
    if len(list1)==1:
        for j in range(len(pd)):       
            if j == list1[0]:
                [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 1)
                pdtmp_list1.append(Conv_update)
                tmp = p
                ptnew.extend([p,p,p,p,p])
            if j in list2:
                [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 2,r_channel = tmp)
                pdtmp_list2.append(Conv_update)
                tmp = p
    if len(list1)==2:
        for j in range(len(pd)):  
            if j == list1[0]:
                [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 1)
                pdtmp_list1.append(Conv_update)
                tmp = p
                ptnew.extend([p,p,p,p,p])
            if j == list1[1]:
                [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 3,r_channel = tmp)
                pdtmp_list1.append(Conv_update)
                tmp = p
                ptnew.extend([p,p,p,p,p])
            if j in list2:
                [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 2,r_channel = tmp)
                pdtmp_list2.append(Conv_update)
                tmp = p
        
    #stati = ([16,16,16])
    
    #ptnew = [ptmp[0]]*5+[ptmp[10]]*5
    #pca_update_layers = ptnew
    renew_model = build_model(np.array(ptnew))
    
    
    #if len(list1)==1:
     #   for k in range(len(pd)):
      #      if k == list1[0]:
       #         renew_model.layers[k].set_weights(pdtmp_list1[0])
        #    if k in list2:
         #       renew_model.layers[k].set_weights(pdtmp_list2[list2.index(k)])
    #if len(list1)==2:
     #   for k in range(len(pd)):
      #      if k == list1[0]:
       #         renew_model.layers[k].set_weights(pdtmp_list1[0])
        #    if k == list1[1]:
         #       renew_model.layers[k].set_weights(pdtmp_list1[1])
          #  if k in list2:
           #     renew_model.layers[k].set_weights(pdtmp_list2[list2.index(k)])
    
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

