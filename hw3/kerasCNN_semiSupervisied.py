import pickle, pprint
import keras
import numpy
import h5py
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.utils import np_utils
from numpy import zeros, ones

#path
trainDataFileName = 'data/all_label.p'
unlabelDataFileName = 'data/all_unlabel.p'
testDataFileName = 'data/test.p'

#parameter
n_train = 5000
n_test = 10000
n_ulabel = 45000
n_classes = 10
channelNumber = 3
width = 32
height = 32

data_augmentation = True

def initKerasModel():
    model = Sequential()
    return model

def loadTrainData():

    X_train_bre = zeros( shape = ( n_train, 3072 ) )
    Y_train_bre = zeros( shape = ( n_train, n_classes ) )
    X_train = zeros( shape = ( n_train, 3072 ) )
    Y_train = zeros( shape = ( n_train, n_classes ) )
    all_label = pickle.load( open( trainDataFileName, 'rb' ) )

    for y in range( n_classes ):
        for x in range( 500 ):
            X_train_bre[ y*500+x, : ] = all_label[y][x]
            Y_train_bre[ y*500+x, y ] = 1

    #shuffle data
    index_shuf = range( n_train )
    random.shuffle( index_shuf )
    for i in range( 0, len( index_shuf )):
        X_train[i] = X_train_bre[index_shuf[i]] 
        Y_train[i] = Y_train_bre[index_shuf[i]] 

    #reshape 2D array to 4D array (for CNN)
    X_train = X_train.reshape( n_train, channelNumber, height, width )
    
    #normalize from [ 0, 255 ] to [ 0, 1 ]
    X_train /= 255
    
    return X_train, Y_train

def updateTrainData( X_train, Y_train, addData, result, n ):

    #update train data
    oriTrainData = (X_train.shape)[0]
    newX_train = zeros( shape = ( oriTrainData + n , channelNumber, height, width ) )
    newY_train = zeros( shape = ( oriTrainData + n, n_classes ) )
    newX_train[ : oriTrainData, : , : , : ] = X_train
    newX_train[ oriTrainData : oriTrainData + n, : , : , : ] = addData
    newY_train[ : oriTrainData, : ] = Y_train
    newY_train[ oriTrainData : oriTrainData + n, :  ] = result

    X_train = zeros( shape = ( (newX_train.shape)[0], channelNumber, height, width ) )
    Y_train = zeros( shape = ( (newX_train.shape)[0], n_classes ) )

    #shuffle data
    index_shuf = range( (newX_train.shape)[0] )
    random.shuffle( index_shuf )
    for i in range( 0, len( index_shuf )):
        X_train[i] = newX_train[index_shuf[i]] 
        Y_train[i] = newY_train[index_shuf[i]]  

    model.fit( X_train, Y_train, nb_epoch= 50, batch_size = 32, validation_split = 0.1, shuffle = True )

    return X_train, Y_train

def selfTraining( model, mode, X_train, Y_train ):

    data = pickle.load( open( unlabelDataFileName, 'rb' ) )
    if mode == 0 : 
        #retrain model after adding 15000 datas
        for i in range( 0, n_ulabel, 15000 ):
            addData = zeros( shape = ( 15000, 3072 ) )
            #predict data
            for x in range( i, i+15000 ):
                addData[x-i] = data[x]
            addData = addData.reshape( 15000, channelNumber, height, width )
            addData /= 255
            result = model.predict_classes( addData )
            result = np_utils.to_categorical( result )

            X_train, Y_train = updateTrainData( X_train, Y_train, addData, result, 15000 )           
    elif mode == 1 :
        #retrain model after adding 5000 datas
        threshold = 0.7
        addData = zeros( shape = ( 15000, channelNumber, height, width ) )
        addLabel = zeros( shape = ( 15000, n_classes ) )
        addDataCount = 0
        for i in range( n_ulabel ):
            tmp = zeros( shape = ( 1, 3072 ) )
            tmp[ 0, : ] = data[i]
            tmp = tmp.reshape( 1, channelNumber, height, width )
            tmp /= 255
            result = model.predict_proba( tmp )

            maxValue = 0
            maxIdx = 0
            for x in range( n_classes ):
                if result[ 0, x] > maxValue:
                    maxValue = result[ 0, x]
                    maxIdx = x

            if maxValue > threshold:
                addData[ addDataCount ] = tmp
                addLabel[ addDataCount, maxIdx ] = 1
                addDataCount += 1

            if addDataCount == 15000:
                X_train, Y_train = updateTrainData( X_train, Y_train, addData, result, 15000 )
                addDataCount = 0
                addData = zeros( shape = ( 15000, channelNumber, height, width ) )
                addLabel = zeros( shape = ( 15000, n_classes ) )

        subAddData = zeros( shape = ( addDataCount, channelNumber, height, width ) )
        subAddData = addData[ : addDataCount ]
        X_train, Y_train = updateTrainData( X_train, Y_train, subAddData, result, addDataCount )

    return model


model = initKerasModel()
X_train, Y_train = loadTrainData()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape= ( channelNumber, height, width ) , dim_ordering = "th" ) )
model.add(LeakyReLU(alpha=.01))
model.add(Convolution2D(32, 3, 3, dim_ordering = "th" ) )
model.add(LeakyReLU(alpha=.01))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = "th" ) ) 
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering = "th" ) )
model.add(LeakyReLU(alpha=.01))
model.add(Convolution2D(64, 3, 3, dim_ordering = "th" ) )
model.add(LeakyReLU(alpha=.01))
model.add(MaxPooling2D( pool_size=(2, 2), dim_ordering = "th" ) )
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(LeakyReLU(alpha=.01))
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

X_train = X_train.astype('float32')

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator( datagen.flow(X_train, Y_train,
                        batch_size=128),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=500 )


selfTrainingMode = 0 # 0 -> add all data, 1 -> add a few most confident data
model = selfTraining( model, selfTrainingMode, X_train, Y_train )

scores = model.evaluate( X_train, Y_train )
print("%s : %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('CNN_model_semiSupervisied.h5')
