import mxnet as mx
import numpy as np
import pickle
import cv2
import os.path
import scipy.fftpack as spfft
from sklearn.linear_model import Lasso

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f, encoding='bytes')
    images = dict[b'data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict[b'labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f)
    return dict['label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

imgarray, lblarray = extractImagesAndLabels("cifar-10-batches-py/", "data_batch_2")
#print (imgarray.shape)
#print (lblarray.shape)

categories = extractCategories("cifar-10-batches-py/", "batches.meta")

cats = []
for i in range(0,50):
    saveCifarImage(imgarray[i], "./", "image"+(str)(i))
    category = lblarray[i].asnumpy()
    category = (int)(category[0])
    cats.append(categories[category])
#print (cats)    
    #insert the compressed sensing code here
    #we have extracted 50 images and it is in the folder. try iterating through those images and apply compressed sensing on it
    #save the output in another folder
for id in range(50):
    path='C:/Users/Lakshmi Sridevi/data/'+'image'+str(id)+'.png'
    
    if os.path.exists(path):
        Xorig=cv2.imread(path)
        
        X = cv2.resize(Xorig,(100, 100), interpolation = cv2.INTER_CUBIC)
        #X = cv2.resize(Xorig,(100,100))
        ny,nx = X.shape[:2]
        
        k = round(nx * ny * 0.5) # 50% sample
        ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
        b = X.T.flat[ri]
        b = np.expand_dims(b, axis=1)
        
        A = np.kron(
        spfft.idct(np.identity(nx), norm='ortho', axis=0),
        spfft.idct(np.identity(ny), norm='ortho', axis=0)
        )
        A = A[ri,:]
        lasso = Lasso(alpha=0.001)
        lasso.fit(A, b)
    
        Xat = np.array(lasso.coef_).reshape(nx, ny).T
        
        Xa = idct2(Xat)
        #print (Xa.shape)
        #Xa1=cv2.cvtColor(Xa,cv2.COLOR_GRAY2BGR)
        cv2.imwrite('C:/Users/Lakshmi Sridevi/data/output'+'/'+str(cats[id])+'_'+str(id)+'.png',Xa)
    else:
        print ("Check code")
    
    
    
    
        