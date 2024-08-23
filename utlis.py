import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten,Dense
from tensorflow.keras.optimizers import Adam
def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    coloumns=['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=coloumns)
    #print(data.head)
    #print(data['center'][0])
    #print(getName(data['center'][0]))
    data['Center'] = data['Center'].apply(getName)
    #print(data.head())
    print('Total images imported are',data.shape[0])
    return data
def balanceData(data,display=True):
    nBins=31
    samplesPerBin=350
    hist,bins=np.histogram(data['Steering'],nBins)
    #print(bins)
    if display:
        center=(bins[:-1]+ bins[1:])*0.5
        #print(center)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
        removeIndexList=[]
        for j in range(nBins):
            binDataList=[]
            for i in range(len(data['Steering'])):
                if data['Steering'][i]>=bins[j] and data['Steering'][i]<=bins[j+1]:
                    binDataList.append(i)
            binDataList = shuffle(binDataList)
            binDataList=binDataList[samplesPerBin:]
            removeIndexList.extend(binDataList)
        print('Removed Images:',len(removeIndexList))
        data.drop(data.index[removeIndexList],inplace=True)
        print('Remaining Images:', len(data))
        if display:
            hist,_ = np.histogram(data['Steering'], nBins)
            plt.bar(center, hist, width=0.06)
            plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
            plt.show()
    return data

def loadData(path,data):
    imagePath=[]
    steering=[]
    for i in range (len(data)):
        indexedData=data.iloc[i]
        imagePath.append(os.path.join(path,'IMG',indexedData[0]))
        #print(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))
    imagePath=np.asarray(imagePath)
    steering=np.asarray(steering)
    return imagePath,steering
def augmentImage(imgPath,steering):
    #PAN
    img=mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    #ZOOM
    if np.random.rand() < 0.5:
        zoom=iaa.Affine(scale=(1,1.2))
        img=zoom.augment_image(img)

    #BRIGTHNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply(0.4,0.9)
        img = brightness.augment_image(img)
    #FLIP
    if np.random.rand() < 0.5:
        img=cv2.flip(img,1)
        steering=-steering
    return img, steering
def preProcessing(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    return img
def batchGen(imagePath,steeringlist,batchsize,trainFlag):
    while True:
        imgBatch=[]
        steeringBatch=[]
        for i in range(batchsize):
            index=random.randint(0,len(imagePath)-1)
            if trainFlag:
                img,steering=augmentImage(imagePath[index],steeringlist[index])
            else:
                img=mpimg.imread(imagePath[index])
                steering=steeringlist[index]
            img=preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asarray(steeringBatch))
def createModel():
    model=Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=0.0001),loss='mse')

    return model