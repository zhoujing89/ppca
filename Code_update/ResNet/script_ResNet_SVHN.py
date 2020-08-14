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
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import add,Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D,BatchNormalization,Input, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import Model, load_model
from keras import optimizers, regularizers
from keras.optimizers import Adam,SGD
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from scipy.io import loadmat as load

from pandas.core.frame import DataFrame
from PIL import Image
from sklearn.decomposition import PCA


session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,allow_soft_placement=True)
session_conf.gpu_options.allow_growth=True                      #不全部占满显存, 按需分配 
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--percentage", type=float, default=0.85, help="累计方差贡献率占比")
parser.add_argument("-u", "--update", type=int, default=1, help="更新次数")
parser.add_argument("-s", "--start", type=int, default=3, help="开始层数")
parser.add_argument("-m", "--blocklayer", type=int, default=1, help="开始的block")
args = parser.parse_args()



## 控制随机性
seed = 2020
np.random.seed(seed) # seed是一个固定的整数即可
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)

date = '0729'

# 载入SVHN数据集
traindata = load('/home/pkustudent/notebooks/SVHN/train_32x32.mat')
testdata = load('/home/pkustudent/notebooks/SVHN/test_32x32.mat')


def reformat(samples, labels):
    # 改变原始数据的形状
    # (图片高，图片宽，通道数，图片数)->(图片数,图片高，图片宽，通道数)
    # labels 变成one-hot encoding
    samples = np.transpose(samples, (3, 0, 1, 2))
    labels = np.array([x[0] for x in labels])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return samples, labels

train_samples = traindata['X']
train_labels = traindata['y']
test_samples = testdata['X']
test_labels = testdata['y']
 
X0,YY0 = reformat(train_samples, train_labels)
X1,YY1 = reformat(test_samples, test_labels)

X0 = X0/255.0
X1 = X1/255.0


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
## def a ResNet32 model

## n=5时是32层
## n=9时是56层
stack_n            = 5
layers             = 6 * stack_n + 2
num_class           = 10
batch_size         = 128
epochs             = 200
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4


def build_model(datlist):
    # input: 32x32x3 output: 32x32x16
    IMSIZE = 32
    input_layer = Input([IMSIZE,IMSIZE,3])
    x = input_layer
    x = Conv2D(datlist[0], (3, 3), strides=(1,1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
               kernel_initializer="he_normal")(x)

    # res_block1 to res_block5 input: 32x32x16 output: 32x32x16
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
    
    # res_block6 input: 32x32x16 output: 16x16x32
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
    
    # res_block7 to res_block10 input: 16x16x32 output: 16x16x32
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
        
    # res_block11 input: 16x16x32 output: 8x8x64
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

    # res_block12 to res_block15 input: 8x8x64 output: 8x8x64
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
    
    # Dense input: 8x8x64 output: 64
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(num_class,activation='softmax',kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    output_layer=x
    model=Model(input_layer,output_layer)
    return model


def scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001

sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
change_lr = LearningRateScheduler(scheduler)


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


renew_model = load_model('/home/pkustudent/notebooks/full_model/ResNet_SVHN_full.h5')  #加载已经保存好的模型


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
    pdtmp_list1 = []  ## 存储中间更新的权重列表
    pdtmp_list2 = []
    #pdtmp_list3 = []
    #ptmp = []   ## 存储中间层降的维度
    ptnew = ori_layer[:args.start]
    
    if len(list1)==1:
        for j in range(len(pd)):       
            if j == list1[0]:            ## 更新其他卷积层
                [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 1)
                pdtmp_list1.append(Conv_update)
                tmp = p
                ptnew.extend([p,p,p,p,p])
            if j in list2:
                [Conv_update,p] =  kernel_channel_wise_pca_conv(pd[j],mode = 2,r_channel = tmp)
                pdtmp_list2.append(Conv_update)
                #ptmp.append(p)
                tmp = p
    if len(list1)==2:
        for j in range(len(pd)):  
            if j == list1[0]:            ## 更新其他卷积层
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
    #pca_update_layers = ptnew      ## 更新后的模型新输入
    renew_model = build_model(np.array(ptnew))  ## 新的模型结构
    
    
   # if len(list1)==1:
   #     for k in range(len(pd)):
   #         if k == list1[0]:
   #             renew_model.layers[k].set_weights(pdtmp_list1[0])
   #         if k in list2:
   #             renew_model.layers[k].set_weights(pdtmp_list2[list2.index(k)])
   # if len(list1)==2:
   #     for k in range(len(pd)):
   #         if k == list1[0]:
   #             renew_model.layers[k].set_weights(pdtmp_list1[0])
   #         if k == list1[1]:
   #             renew_model.layers[k].set_weights(pdtmp_list1[1])
   #         if k in list2:
   #             renew_model.layers[k].set_weights(pdtmp_list2[list2.index(k)])
    
    renew_model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics = ['accuracy'])
    renew_model.summary()
    renew_model.fit(datagen.flow(X0, YY0,batch_size=batch_size),
                     steps_per_epoch=iterations,
                     epochs=1,
                     callbacks=[change_lr],
                     validation_data=(X1, YY1),verbose =1,shuffle=False)
print("=" * 40 )
print(f"finish determine stucture, final train...")
print("=" * 40 )
    
filepath_1 = f'/home/pkustudent/notebooks/PPCA_update/PPCA_update/ResNet/SVHN/ResNet-SVHN-percentage{args.percentage}-update{args.update}-start{args.start}-' + date

os.mkdir(filepath_1)

filepath = filepath_1 + '/weights-improvement-{epoch:02d}-{val_accuracy:.4f}.h5'

#filepath="/home/pkustudent/notebooks/PPCA_update/PPCA_update/AlexNet/Cifar10/PPCA_best_model_result/AlexNet-cifar10-percentage{args.percentage}-update{args.update}-start{args.start}-weights-improvement-{epoch:02d}-{val_accuracy:.4f}.h5"

checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
 
history = renew_model.fit(datagen.flow(X0, YY0, batch_size=batch_size),
                          steps_per_epoch=X0.shape[0] // batch_size,
                          epochs=epochs,
                          validation_data=(X1,YY1),
                          verbose =1,
                          shuffle=False,
                          callbacks=[checkpoint,change_lr])