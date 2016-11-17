import keras
import pickle
import csv
import sys
import numpy
from keras.models import load_model
from numpy import zeros


inputFilePath = sys.argv[1]
modelName = sys.argv[2] + '.h5'
outputName = sys.argv[3]
testDataFileName = inputFilePath + 'test.p'

n_test = 10000
n_class = 10
channelNumber = 3
height = 32
width = 32


NORMALIZE = True

def loadTestData():

	X_test = zeros( shape = ( n_test, 3072 ) )

	test = pickle.load( open( testDataFileName, 'rb' ) )

	for x in range( n_test ):
		X_test[ x, : ] = test['data'][x]

	#reshape 2D array to 4D array (for CNN)
	X_test = X_test.reshape( n_test, channelNumber, height, width )

	#normalize from [ 0, 255 ] to [ 0, 1 ]
	if NORMALIZE == True:
		X_test /= 255

	return X_test

def writeCSV( predictions ):

	result = [['ID', 'class']]
	for x in range( n_test ):
		r = []
		r.append(x)
		r.append(predictions[x])
		result.append(r)


	f = open( outputName, 'w' )
	w = csv.writer(f)
	w.writerows(result)
	f.close()


#load model
model = load_model(modelName)

X_test = loadTestData()
predictions = model.predict_classes( X_test )
writeCSV( predictions )

