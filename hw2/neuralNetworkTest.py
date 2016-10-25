import numpy as np
from numpy import ones, zeros, mean, std
import csv
import sys
import math

#get sys argument
inputModelName = sys.argv[1]
testDataName = sys.argv[2]
outputName = sys.argv[3]

#parameter
featureNum = 57
outputNode = 1
hidddenNeuron = 10
inputNode = 57 

def sigmoid(x):
	return math.tanh(x)


def dsigmoid(y):
	return 1.0 - y**2


def loadModel():

	model = np.load( inputModelName )
	mean_r = model[ 0, : ]
	std_r = model[ 1, : ]
	wi = model[ 2 : 60, : hidddenNeuron+1 ]  #size 57*11
	wo = model[ 60 : , : 1 ]

	wi.shape = ( featureNum+1, hidddenNeuron+1 )
	wo.shape = ( hidddenNeuron+1, 1 )

	return mean_r, std_r, wi, wo


def loadTestData(fileName):

	dataList = []
	count = 0

	f = open( fileName, 'r', encoding = "ISO-8859-1" )
	for row in csv.reader(f):
		data = []
		for i in range( 1, len(row) ):
			data.append( row[i] )

		dataList.append(data)
		count = count + 1

	dataList = np.matrix( dataList, dtype = np.float64 )
	return dataList, count


def update( inputs ):

	ni = inputNode+1
	nh = hidddenNeuron+1
	no = outputNode

	#declare activation for nodes
	ai = ones( shape = ( ni, 1 ) )
	ah = ones( shape = ( nh, 1 ) )
	ao = ones( shape = ( no, 1 ) )

	if (inputs.shape)[0] != ni-1:
		print("wrong number of inputs")

	#input activations
	ai[ :ni-1  , : ] = inputs[ : ni-1, :  ]

	#hidden activations
	for j in range( nh -1 ):
		total = 0.0
		for i in range( ni ):
			total += ai[ i, 0 ]*wi[ i, j ]
		ah[ j, 0 ] = sigmoid(total)

	# print("ah = ")
	# print(self.ah)

	#output activations
	for k in range( no ):
		total = 0.0
		for j in range( nh ):
			total += ah[ j, 0 ] * wo[ j, k ]
		ao[k] = sigmoid(total)

	return ao

def test(  testData, wi, wo ):

	result = [['id', 'label']]

	m = (testData.shape)[0]

	for x in range(m):

		r = []

		tmp = testData[ x, : ]
		tmp.shape = ( featureNum, 1 )
		res = update(tmp)

		if res < 0.5:
			res = 0
		else:
			res = 1

		r.append(x+1)
		r.append(int(res))
		result.append(r)

	return result


def testDataNormalization( testData,  mean_r, std_r ):

	for x in range( featureNum ):
		testData[ :, x ] = ( testData[ :, x] - mean_r[x] ) / std_r[x]
	return testData



def writeCSV( result ):
	f = open( outputName, 'w' )
	w = csv.writer(f)
	w.writerows(result)
	f.close()


#######Load test data
mean_r, std_r, wi, wo = loadModel()

#######Testing
testData, testDataCount = loadTestData( testDataName )
testData = testDataNormalization( testData, mean_r, std_r )
result = test( testData, wi, wo )
writeCSV(result) 