from __future__ import print_function

#import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from time import time
import matplotlib.pyplot as plt

#print('tensorflow:', tf.__version__)
#print('tensorflow.keras:', tensorflow.keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8

#To input our values in our network Dense layer, we need to flatten the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 784)
#flatten images
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train)
y_test = tensorflow.keras.utils.to_categorical(y_test)


num_classes = y_train.shape[1]
#Faire plus de 0.99, 0.98 c'est bien
#on peut ajouter des couches
#courbe train test sur meme image
results = []

for bs in [128,200,500]:
    model = Sequential()
    model.add(Dense(64,input_dim = 784,activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10,activation = 'sigmoid'))
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics=['accuracy'])
    start = time()
    hist = model.fit(x_train,y_train,epochs = 100,batch_size = bs,validation_data = (x_test,y_test))
    plt.plot(range(100),hist.history['val_accuracy'],'go--', linewidth=2, markersize=12)
    plt.show()
    end = time() - start
    _,accuracy = model.evaluate(x_test,y_test)
    results.append("Batch size:"+str(bs)+" Accuracy:"+str(accuracy)+" Time:"+str(end))

for r in results:
    print(r)

