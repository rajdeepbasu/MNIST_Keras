# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.regularizers import l2 # L2-regularisation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Reading training and test dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

##Preprocessing the data

# Changing pixel values to float
train_x = train.iloc[:,1:].values.astype("float32")
train_y = train.iloc[:,:1].values.astype("float32")
test = test.iloc[:,0:].values.astype("float32")

#Converting label to categorical
train_y = to_categorical(train_y)

#Standardizing/Normalizing the dataset
train_x = preprocessing.normalize(train_x)
test = preprocessing.normalize(test)

#Reshaping the dataset
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test = test.reshape(test.shape[0], 28, 28, 1)

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state=random_seed)

## Defining the convolutional network

#Initializing the CNN
classifier = Sequential()

# Adding a convolutional layer
classifier.add(Conv2D(64,(3,3), padding = 'Same',activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.0001), input_shape = (28,28,1)))
classifier.add(Conv2D(64,(3,3), padding = 'Same',activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.25))

# Adding a second convolutional layer
classifier.add(Conv2D(64,(3,3),padding = 'Same',activation ='relu'))
classifier.add(Conv2D(64,(3,3),padding = 'Same',activation ='relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.25))

# Adding a fully connected layer
classifier.add(Flatten())
classifier.add(Dense(128, activation = "relu"))
classifier.add(Dense(128, activation = "relu"))
classifier.add(Dropout(0.2))
classifier.add(Dense(10, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0001), activation = "softmax"))

# Compiling the CNN
classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Setting early stopping criteria
EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

# Creating an imagedatagenator to create additional training dataset
Image_Gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
Image_Gen.fit(train_x)

train_generator = Image_Gen.flow(train_x,train_y, batch_size=128)

model = classifier.fit_generator(train_generator,
                         epochs = 25, validation_data = (val_x,val_y),
                         verbose = 1, steps_per_epoch=train_x.shape[0]//128,callbacks=[EarlyStop]

# list all data in history
print(model.history.keys())
# summarize history for accuracy
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

preds = classifier.predict_classes(test, verbose=0)
output = pd.DataFrame(
    {   "ImageId": list(range(1,len(preds)+1)),
        "Label": preds
    }
)
output.to_csv("outcome_modified.csv", index=False, header=True)
