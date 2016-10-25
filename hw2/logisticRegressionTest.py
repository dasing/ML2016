import numpy as np
import sys
import csv


#parameter
featureNum = 57

#get sys argument
inputModelName = sys.argv[1]
testDataName = sys.argv[2]
outputName = sys.argv[3]


def loadModel():

	model = np.load( inputModelName )
	mean_r = model[ 0, : ]
	std_r = model[ 1, : ]
	weight = model[ 2, : ]
	weight.shape = ( featureNum+1, 1 )

	return mean_r, std_r, weight


def sigmoid( X ):

	d = 1.0/(1.0 + np.exp( -1.0*X))
	return d


def loadTestData(fileName):

	dataList = []
	count = 0

	f = open( fileName, 'r', encoding = "ISO-8859-1" )
	for row in csv.reader(f):
		data = []
		for i in range( 1, len(row) ):
			data.append( row[i] )

		data.append(1.0)
		dataList.append(data)
		count = count + 1

	dataList = np.matrix( dataList, dtype = np.float64 )
	return dataList, count

def testDataNormalization( testData,  mean_r, std_r ):

	for x in range( featureNum ):
		testData[ :, x ] = ( testData[ :, x] - mean_r[x] ) / std_r[x]

	return testData

def computeTestDataResult( testData, weight ):

	result = [['id', 'label']]

	prediction = sigmoid( testData.dot(weight) )
	m = prediction.size

	for x in range( m ):
		r = []
		if prediction.item(x) < 0.5:
			prediction[ x, 0 ] = 0
		else:
			prediction[ x, 0 ] = 1

		r.append(x+1)
		r.append(int(prediction.item(x)))
		result.append(r)

	return result

def writeCSV( result ):
	f = open( outputName, 'w' )
	w = csv.writer(f)
	w.writerows(result)
	f.close()

#######Load test data
mean_r, std_r, weight = loadModel()

######Testing
testData, testDataCount = loadTestData(testDataName)
testData = testDataNormalization( testData, mean_r, std_r )
result = computeTestDataResult( testData, weight )
writeCSV(result)