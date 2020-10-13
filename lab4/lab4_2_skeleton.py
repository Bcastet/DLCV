from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Softmax
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np
from time import time


print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)


##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec) 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tensorflow.keras.utils.to_categorical(y_train,10)
y_test = tensorflow.keras.utils.to_categorical(y_test,10)
#Let start our work: creating a convolutional neural network

#####TO COMPLETE
#0.75 = Le boss
#0.7 c'est bien
N = 3
M = 2
K = 3

results = []

model = Sequential()
for i in range(M):
    for i in range(N):
        model.add(Conv2D(32,kernel_size=(3,3),strides = (1,1),input_shape = (32,32,3),padding = 'same',activation='relu'))
    model.add(MaxPooling2D(strides = (1,1),pool_size=(2,2)))
    model.add(Dropout(0.3))
model.add(Flatten())

for i in range(K):
    model.add(Dense(64,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))
print("N:"+str(N)+" M:"+str(M)+" K:"+str(K))
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics=['accuracy'])
start = time()
model.fit(x_train,y_train,epochs = 50,batch_size = 128,validation_data = (x_test,y_test))
end = time() - start
_,accuracy = model.evaluate(x_test,y_test)
results.append("N:"+str(N)+" M:"+str(M)+" K:"+str(K))
                        

for r in results:
    print(r)

