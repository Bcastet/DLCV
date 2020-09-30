from __future__ import print_function
import numpy as np

#In this first part, we just prepare our data (mnist)
#for training and testing

#import keras
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels = X_train.shape[1] * X_train.shape[2]
#On appliatis nos images
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255 #normalisation
X_test  = X_test / 255 #normalisation


#We want to have a binary classification: digit 0 is classified 1 and
#all the other digits are classified 0
#Notre réseau de neurone va reconnaitre les 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new

#On fait les transposées
y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

# #Display one image and corresponding label
# import matplotlib
# import matplotlib.pyplot as plt
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()


#Let start our work: creating a neural network
#First, we just use a single neuron.


def sigmoid(z):
    return 1./(1 + np.exp(-z))


def compute_loss(y, yhat):
    l = -1. / m * (
        np.sum(np.multiply(y, np.log(yhat))) +
        np.sum(np.multiply(1 - y, np.log(1 - yhat))))
    return l

X = X_train
Y = y_train

n = X.shape[0]
m = X.shape[1]

#On initialise aléatoirement par une loi centrée-reduite
W = np.random.randn(1,n) * 0.01
b = np.zeros((1,1))

lr = 0.01 #learning rate
epochs = 500
for i in range(epochs):
    #premier layer
    Z = np.matmul(W, X) + b
    Yhat = sigmoid(Z)

    loss = compute_loss(Y, Yhat)

    #on calcule la dérivée dL/dW2
    dZ = Yhat - Y
    dW = 1. / m * np.matmul(dZ, X.T)
    #on calcule la dérivée dL/db2
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)

    #descente de gradient
    W = W - lr * dW
    b = b - lr * db

    print('Epoch ',i, ' loss value : ', loss)

Z = np.matmul(W,X_test) + b
Yhat = sigmoid(Z)

loss = compute_loss(y_test,Yhat)
print("Final accuracy:",1-loss)
