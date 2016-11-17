import pickle, pprint
import keras
import numpy
import h5py
import random
import sys
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, UpSampling2D, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from numpy import zeros, ones

inputFilePath = sys.argv[1]
outputName = sys.argv[2]
encdodedModelName = outputName + '.h5'
classModelName = outputName + '_2.h5'

#path
trainDataFileName = inputFilePath + 'all_label.p'
unlabelDataFileName = inputFilePath + 'all_unlabel.p'
testDataFileName = inputFilePath + 'test.p'

#parameter
n_train = 5000
n_ulabel = 45000
n_test = 10000
n_classes = 10
channelNumber = 3
width = 32
height = 32

#conv parameter
n_filters = 64
n_conv = 3 #filter window size
n_pool = 2 #pooling window size


#function parameter
NORMALIZE = True


def constructAutoEncoderModel():

	input_img = Input( shape = ( channelNumber, height, width ) )

	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering="th" )(input_img)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering="th")(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering="th" )(x)
	encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering="th" )(x)

	## dimension( 16 * 8 * 8 )

	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same',dim_ordering="th" )(encoded)
	x = UpSampling2D((2, 2), dim_ordering="th" )(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering="th" )(x)
	x = UpSampling2D((2, 2), dim_ordering="th" )(x)
	
	decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', dim_ordering="th" )(x)

	autoencoder = Model( input_img, decoded )
	autoencoder.compile( optimizer='adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'] )

	return input_img, encoded, decoded, autoencoder

def constructAutoEncodedModel( input_img, encoded, X_train ):

	encodedModel = Model( input_img, encoded )

	code = encodedModel.predict( X_train )

	#print( code.shape ) ( 5000, 16, 4, 4 )

	return encodedModel, code


def loadTrainData():

	X_train_bre = zeros( shape = ( n_train + n_ulabel, 3072 ) )
	Y_train_bre = zeros( shape = ( n_train, n_classes ) )
	X_train_mix = zeros( shape = ( n_train + n_ulabel, 3072 ) )
	X_train = zeros( shape = ( n_train, 3072 ) )
	Y_train = zeros( shape = ( n_train, n_classes ) )
	all_label = pickle.load( open( trainDataFileName, 'rb' ) )
	all_ulabel = pickle.load( open( unlabelDataFileName, 'rb' ) )

	for y in range( n_classes ):
		for x in range( 500 ):
			X_train_bre[ y*500+x, : ] = all_label[y][x]
			Y_train_bre[ y*500+x, y ] = 1

	for x in range( n_ulabel ):
		X_train_bre[ n_train + x, : ] = all_ulabel[x]

	#shuffle data
	index_shuf = range( n_train + n_ulabel )
	random.shuffle( index_shuf )
	count = 0
	for i in range( 0, len( index_shuf )):
		if index_shuf[i] < n_train :
			X_train[count] = X_train_bre[index_shuf[i]] 
			Y_train[count] = Y_train_bre[index_shuf[i]]
			count += 1
			
		X_train_mix[i] = X_train_bre[index_shuf[i]]
		
	#reshape 2D array to 4D array (for CNN)
	X_train = X_train.reshape( n_train, channelNumber, height, width )
	X_train_mix = X_train_mix.reshape( n_train + n_ulabel, channelNumber, height, width )
	
	#normalize from [ 0, 255 ] to [ 0, 1 ]
	if NORMALIZE == True:
		X_train /= 255
		X_train_mix /= 255
	
	return X_train, Y_train, X_train_mix


def updateTrainData( X_train, Y_train, addData, result, n, model ):

	#update train data
	oriTrainData = (X_train.shape)[0]
	newX_train = zeros( shape = ( oriTrainData + n , 16, 8, 8 ) )
	newY_train = zeros( shape = ( oriTrainData + n, n_classes ) )
	newX_train[ : oriTrainData, : , : , : ] = X_train
	newX_train[ oriTrainData : oriTrainData + n,  : , : , : ] = addData
	newY_train[ : oriTrainData, : ] = Y_train
	newY_train[ oriTrainData : oriTrainData + n, :  ] = result

	X_train = zeros( shape = ( (newX_train.shape)[0], 16, 8, 8 ) )
	Y_train = zeros( shape = ( (newX_train.shape)[0], n_classes ) )

	#shuffle data
	index_shuf = range( (newX_train.shape)[0] )
	random.shuffle( index_shuf )
	for i in range( 0, len( index_shuf )):
		X_train[i] = newX_train[index_shuf[i]] 
		Y_train[i] = newY_train[index_shuf[i]] 	

	model.fit( X_train, Y_train, nb_epoch= 50, batch_size = 128, validation_split = 0.1, shuffle = True )

	return X_train, Y_train


def codeClassify( code, Y_train ):

	print( code.shape )
	#ode = code.reshape( n_train, 1024 ) # origin code shpae = ( 5000, 16, 8, 8 )

	classModel = Sequential()

	classModel.add(Convolution2D(32, 3, 3, border_mode='same', input_shape= ( 16, 8, 8 ) , dim_ordering = "th" ) )
	classModel.add(Activation('relu'))
	classModel.add(Convolution2D(32, 3, 3, dim_ordering = "th" ) )
	classModel.add(Activation('relu'))
	classModel.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = "th" ) ) 
	classModel.add(Dropout(0.25))
	classModel.add(Flatten())

	classModel.add(Dense( 512, activation='relu'))
	classModel.add(Dropout(0.5))
	classModel.add(Dense( n_classes, activation='softmax') )

	# Compile model
	classModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	classModel.fit( code, Y_train, nb_epoch = 100, batch_size = 128, shuffle = True )

	return classModel, code


def codeClassifySelfTraining( classModel, encodedModel, X_train, Y_train ):

	data = pickle.load( open( unlabelDataFileName, 'rb' ) )	

	for i in range( 0, n_ulabel, 15000 ):

		#print("i =", i )
		addData = zeros( shape = ( 15000, 3072 ) )

		#print("i =", i, "1" )

		#predict data
		curr = i
		#print("i =", i, "1.5" )
		for x in range( curr, curr+15000 ):
			#print("i =", i, x, " haha !" )
			addData[x-curr] = data[x]

		#print("i =", i, "2" )

		addData = addData.reshape( 15000, channelNumber, height, width )
		addData /= 255

		#print("i =", i , "3" )
		encodedAddData = encodedModel.predict( addData )
		#encodedAddData = encodedAddData.reshape( 15000, 1024 )

		#print("i =", i , "4" )

		predictResult = classModel.predict_classes( encodedAddData )
		predictResult = np_utils.to_categorical( predictResult, nb_classes = n_classes )

		X_train, Y_train = updateTrainData( X_train, Y_train, encodedAddData, predictResult, 15000, classModel )
		#print("i =", i, "return" )

	return classModel	



input_img, encoded, decoded, autoencoder = constructAutoEncoderModel()
X_train, Y_train, X_train_mix = loadTrainData()

#train auto-encoder
autoencoder.fit( X_train, X_train, nb_epoch = 100, batch_size = 128, shuffle = True )

#get encoded model
encodedModel, code = constructAutoEncodedModel( input_img, encoded, X_train )
classModel, code = codeClassify( code, Y_train )
classModel = codeClassifySelfTraining( classModel, encodedModel, code, Y_train )


encodedModel.save(encdodedModelName)
classModel.save(classModelName)
