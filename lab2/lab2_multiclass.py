from __future__ import print_function
import numpy as np

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



# one-hot encode labels
digits = 10

def one_hot_encode(y, digits):
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int32')]  #shape (1, 70000, 10)
    Y_new = Y_new.T.reshape(digits, examples)
    return Y_new

y_train=one_hot_encode(y_train, digits)
y_test=one_hot_encode(y_test, digits)

#Now, we shuffle the training set
np.random.seed(138)
m = X_train.shape[1]
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_loss(y,y_hat):
    return -np.sum(np.multiply(y,np.log(Yhat)))

X = X_train
Y = y_train

n = X_train.shape[0]

lr = 1
nh = 64

W1 = np.random.randn(nh,n) 
b1 = np.zeros((nh,1))

W2 = np.random.randn(10,nh) 
b2 = np.zeros((10,1))

epochs = 150

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
accurate =0

y_test = y_test.T
Yhat = Yhat.T

for i in range(len(y_test)):
    if np.argmax(y_test[i]) == np.argmax(Yhat[i]):
        accurate+=1

print("Final accuracy:",accurate/len(y_test))



