from __future__ import print_function
import numpy as np
import math
#In this first part, we just prepare our data (mnist) 
#for training and testing

#import keras
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0



y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==3.)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==3.)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_loss(y,y_hat):
    l = - 1./m*(np.sum(np.multiply(y,np.log(y_hat))) + np.sum(np.multiply(1-y,np.log(1-y_hat)))) 
    return l

X = X_train
Y = y_train

n = X_train.shape[0]

lr = 0.01
nh = 64

W1 = np.random.randn(nh,n) * 0.01
b1 = np.zeros((nh,1))

W2 = np.random.randn(1,nh) * 0.01
b2 = np.zeros((1,1))

epochs = 100

for i in range(epochs):
    Z1 = np.matmul(W1,X) + b1
    Y1 = sigmoid(Z1)

    Z2 = np.matmul(W2,Y1) + b2
    Yhat = sigmoid(Z2)

    loss = compute_loss(Y,Yhat)

    dZ2 = Yhat - Y
    dW2 = 1. / m * np.matmul(dZ2, Y1.T)
    db2 = 1. / m * np.sum(dZ2, axis = 1, keepdims = True)

    dY1 = np.matmul(W2.T,dZ2)
    dZ1 = dY1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = 1. / m* np.matmul(dZ1,X.T)
    db1 = 1. / m * np.sum(dZ1,axis = 1, keepdims=True)

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2 
    
    if i%10==0:
        print("Epoch",i,"loss value:",loss)


Z1 = np.matmul(W1,X_test) + b1
Y1 = sigmoid(Z1)
Z2 = np.matmul(W2,Y1) + b2
Yhat = sigmoid(Z2)
loss = compute_loss(y_test,Yhat)

print("Final accuracy:",1-loss)


