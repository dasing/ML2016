import math
import random
import numpy as np
from numpy import ones, zeros, mean, std
from math import sqrt
import csv
import sys

#parameter
inputNode = 57
hiddenLayerNum = 10 
outputNode = 1
alpha = 0.1
iteration = 100
featureNum = 57
delta = 0.0000001
M = 0.1

#sys parameter
fileName = sys.argv[1]
modelName = sys.argv[2]


random.seed(0)

def rand( a,b ):
	return (b-a)*random.random() + a

def sigmoid(x):
	return math.tanh(x)

def dsigmoid(y):
	return 1.0 - y**2

class neuralNetwork:
	def __init__( self, ni, nh, no ):

		#Number of input, hideen and output nodes
		self.ni = ni +1 #+1 for bias
		self.nh = nh +1  #+1 for bias
		self.no = no 

		#activation for nodes
		self.ai = ones( shape = ( self.ni, 1 ) )
		self.ah = ones( shape = ( self.nh, 1 ) )
		self.ao = ones( shape = ( self.no, 1 ) )

		#create weights
		self.wi = zeros( shape = ( self.ni, self.nh ) )
		self.wo = zeros( shape = ( self.nh, self.no ) )

		#accumulation gradient
		self.sgi = zeros( shape = ( self.ni, self.nh ) )
		self.sgo = zeros( shape = ( self.nh, self.no ) )

		#momentom
		self.ci = zeros( shape = ( self.ni, self.nh ) )
		self.co = zeros( shape = ( self.nh, self.no ) )

		for y in range( self.ni ):
			for x in range( self.nh ):
				self.wi[ y, x ] = rand( -1, 1 )

		for y in range( self.nh ):
			for x in range( self.no ):
				self.wo[ y, x ] = rand( -1, 1 )

	def update( self, inputs ):

		if (inputs.shape)[0] != self.ni-1:
			print("wrong number of inputs")

		#input activations
		self.ai[ :self.ni-1  , : ] = inputs[ : self.ni-1, :  ]

		#hidden activations
		for j in range( self.nh -1 ):
			total = 0.0
			for i in range( self.ni ):
				total += self.ai[ i, 0 ]*self.wi[ i, j ]
			self.ah[ j, 0 ] = sigmoid(total)

		# print("ah = ")
		# print(self.ah)

		#output activations
		for k in range( self.no ):
			total = 0.0
			for j in range( self.nh ):
				total += self.ah[ j, 0 ] * self.wo[ j, k ]
			self.ao[k] = sigmoid(total)

		return self.ao

	def backPropagate( self, targets ):

		if (targets.shape)[0] != self.no:
			print("wrong number of target values")

		#calculate error terms for output
		outputDeltas = zeros( shape = ( self.no, 1 ) )
		for k in range( self.no ):
			outputDeltas[k, 0] = dsigmoid( self.ao[ k, 0 ] )*( targets[ k, 0 ] - self.ao[ k ,0 ] )

		#calculate error terms for hidden
		hiddenDeltas = zeros( shape = ( self.nh, 1 ) )
		for j in range( self.nh ):
			error = 0.0
			for k in range(self.no):
				error += outputDeltas[ k, 0 ]*self.wo[ j, k ]
			hiddenDeltas[ j, 0 ] = dsigmoid( self.ah[ j, 0] )*error

		#update output weights
		for j in range( self.nh ):
			for k in range( self.no ):
				change = outputDeltas[ k, 0 ]*self.ah[ j, 0 ]
				self.sgo[ j, k ] += change*change
				learningRate0 = alpha/( delta+sqrt(self.sgo[ j, k ] ) )
				self.wo[ j, k ] = self.wo[j, k ] + learningRate0*change + M*self.co[ j, k ]
				#self.wo[ j, k ] = self.wo[j, k ] + alpha*change + M*self.co[ j, k ]
				self.co[ j, k ] = change


		#update input weights
		for i in range( self.ni ):
			for j in range( self.nh ):
				change = hiddenDeltas[j]*self.ai[i]
				self.sgi[ i, j ] += change*change
				learningRate1 = alpha/( delta+sqrt(self.sgi[ i, j ] ) )
				self.wi[i, j] = self.wi[i, j] + learningRate1*change + M*self.ci[ i, j ]
				#self.wo[ j, k ] = self.wo[j, k ] + alpha*change + M*self.co[ j, k ]
				self.ci[ i, j ] = change


		#calculate error
		error = 0.0
		for k in range( len(targets) ):
			if self.ao[k] < 0.5:
				if targets == 1:
					error += 1.0
			else:
				if targets == 0:
					error += 0.0

		return error

	def test( self, testData ):

		result = [['id', 'label']]

		m = (testData.shape)[0]

		for x in range(m):

			r = []

			tmp = testData[ x, : ]
			tmp.shape = ( featureNum, 1 )
			res = self.update(tmp)

			if res < 0.5:
				res = 0
			else:
				res = 1

			r.append(x+1)
			r.append(int(res))
			result.append(r)

		return result


	def train( self, trainData, trainLabel ):

		m = (trainData.shape)[0]
		accumulate = 0

		for i in range( iteration ):
			error = 0.0
			for x in range( m ):
				tmp = trainData[ x, : ] 
				tmp.shape = ( featureNum, 1 )
				self.update( tmp )

				tmpLabel = trainLabel[ x, : ]
				tmpLabel.shape = ( outputNode, 1 )
				tmp = self.backPropagate( tmpLabel )
				error += tmp

			errorRate = error/m
			print( "finish iteraion " + str(i) + ", error rate is " + str(errorRate) )

		return self.wi, self.wo


def loadData(fileName):

	dataList = []
	data = []
	yHead = []
	count = 0

	f = open( fileName, 'r', encoding = "ISO-8859-1" )
	for row in csv.reader(f):
		#print("row size = " + str(len(row))) 
		#print(row)
		data = []
		for i in range( 1, len(row) ):
			if i == len(row)-1:
				yHead.append( float(row[i]) )
				continue
			data.append( float(row[i]) )

		dataList.append(data)
		count = count+1

	label = zeros( shape = (count, 1) )
	for x in range(count):
		label[ x, 0 ] = yHead[x]

	dataList = np.matrix( dataList, dtype = np.float64 )
	print("count = " + str(count) )

	return dataList, label, count

#add bias to dataList
def manageData( dataList, count ):

	trainData = ones( shape = ( count, featureNum) )
	trainData[ :, : featureNum ] = dataList[ :, : ]

	return trainData


def featureNormalization( trainData ):

	mean_r = []
	std_r = []
	trainData_norm = trainData

	for x in range( featureNum ):
		m = mean( trainData[ :, x ] )
		s = std( trainData[ :, x ] )
		mean_r.append(m)
		std_r.append(s)
		trainData_norm[ :, x ] = ( trainData_norm[ :, x ] - m )/s

	return trainData_norm, mean_r, std_r




def outputModel( mean_r, std_r, wi, wo ):

	# wi -> 58*11, wo -> 11*1
	model = np.ones( shape = ( 71, featureNum+1 ) )
	model[ 0, : featureNum ] = mean_r
	model[ 1, : featureNum ] = std_r

	for x in range( 58 ): #store wi
		model[ x+2, : hiddenLayerNum+1 ] = wi[ x, : ]

	for x in range( 11 ):
		model[ 60+x, 0 ] = wo[ x, 0 ]

	fullModelName = modelName + '.npy' 
	np.save( fullModelName, model )

######Training		
dataList, label, count = loadData(fileName)
trainData = manageData( dataList, count )
trainData, mean_r, std_r = featureNormalization( trainData )

NN = neuralNetwork( inputNode, hiddenLayerNum, outputNode )
wi, wo = NN.train( trainData, label )


######Output model
outputModel( mean_r, std_r, wi, wo )




