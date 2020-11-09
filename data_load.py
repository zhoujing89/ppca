#!/usr/bin/env python
# coding: utf-8


################ load CatDog dataset

IMSIZE=32

train_path = '/data/CatDog/train/'
validation_path = '/data/CatDog/validation/'

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    validation_path,
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
    train_path,
    target_size=(IMSIZE, IMSIZE),
    batch_size=batch_size,
    class_mode='categorical')

############### load cifar10 dataset
(X0,Y0),(X1,Y1) = cifar10.load_data()

N0=X0.shape[0];N1=X1.shape[0]
X0 = X0.reshape(N0,32,32,3)/255
X1 = X1.reshape(N1,32,32,3)/255
YY0 = np_utils.to_categorical(Y0)
YY1 = np_utils.to_categorical(Y1)


#mean = np.mean(X0,axis=(0,1,2,3))
#std = np.std(X0,axis=(0,1,2,3))
#X0 = (X0-mean)/(std+1e-7)
#X1 = (X1-mean)/(std+1e-7)

datagen = ImageDataGenerator(
    featurewise_center=False,  
    samplewise_center=False,  
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    shear_range=0.5,
    zoom_range=0.2, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
#datagen.fit(X0)


############## load cifar100 dataset
(X0,Y0),(X1,Y1) = cifar100.load_data(label_mode='fine')

N0=X0.shape[0];N1=X1.shape[0]
X0 = X0.reshape(N0,32,32,3)/255
X1 = X1.reshape(N1,32,32,3)/255


#mean = np.mean(X0,axis=(0,1,2,3))
#std = np.std(X0,axis=(0,1,2,3))
#X0 = (X0-mean)/(std+1e-7)
#X1 = (X1-mean)/(std+1e-7)

YY0 = np_utils.to_categorical(Y0)
YY1 = np_utils.to_categorical(Y1)

datagen = ImageDataGenerator(
   featurewise_center=False,  
   samplewise_center=False,  
   featurewise_std_normalization=False,  
   samplewise_std_normalization=False,  
   zca_whitening=False,  
   rotation_range=15,  
   width_shift_range=0.1,  
   height_shift_range=0.1,  
   horizontal_flip=True,  
   vertical_flip=False)  


############## load SVHN dataset
traindata = load('/train_32x32.mat')
testdata = load('/test_32x32.mat')


def reformat(samples, labels):
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

mean = np.mean(X0,axis=(0,1,2,3))
std = np.std(X0,axis=(0,1,2,3))
X0 = (X0-mean)/(std+1e-7)
X1 = (X1-mean)/(std+1e-7)

datagen = ImageDataGenerator(
    featurewise_center=False,  
    samplewise_center=False,  
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    shear_range=0.5,
    zoom_range=0.2, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

