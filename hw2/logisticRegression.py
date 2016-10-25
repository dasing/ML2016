import numpy as np
import csv
import math
from numpy import ones, zeros, mean, std
from math import sqrt
import time
import sys


#parameter
iteration = 100000
alpha = 1
delta = 0.0000001
featureNum = 57
batchSize = 10

#get sys argument
fileName = sys.argv[1]
modelName = sys.argv[2]

def loadData(fileName):

	dataList = []
	data = []
	yHead = []
	count = 0

	f = open( fileName, 'r', encoding = "ISO-8859-1" )
	for row in csv.reader(f):
		
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



def checkData( dataList, yHead ):

	print(yHead.shape)
	#print("size = " + str((yHead.shape)[0]) )
	for x in range( len(yHead) ):
		print("label = " + str(yHead.item(x) ) )
		print("data = " )
		print( dataList[x] )

#add bias to dataList
def manageData( dataList, count ):

	trainData = ones( shape = ( count, featureNum+1) )
	trainData[ :, : featureNum ] = dataList[ :, : ]

	return trainData

def sigmoid( X ):

	d = 1.0/(1.0 + np.exp( -1.0*X))
	return d

def computeCost( trainData, yHead, weight ):

	m = yHead.size
	#print("m = " + str(m) )
	prediction = sigmoid( trainData.dot(weight) )	

	loss = ((yHead*np.log(prediction) + (1-yHead)*np.log(1-prediction)).sum())/m

	return (-1)*loss

def computeErrorRate( trainData, yHead, weight ):

	m = len(yHead)
	diff = 0
	prediction = sigmoid( trainData.dot(weight) )

	for x in range( m ):
		if prediction.item(x) < 0.5:
			prediction[ x, 0 ] = 0
		else:
			prediction[ x, 0 ] = 1

		if prediction.item(x) != yHead.item(x):
			diff = diff+1

	loss = diff/m

	return loss

def gradientDescent( trainData, yHead, weight, count ):

	J_History = zeros( shape = ( iteration, 1 ) )
	accumulate = 0

	for x in range( 0, iteration ):

		
		prediction =  sigmoid( trainData.dot(weight) )
		

		for i in range( featureNum+1 ):

			
			tmp = trainData[ :, i ]
			tmp.shape = ( count, 1 )
		
			derivative = ( ( ( prediction - yHead )*tmp ).sum() )/count
				
			accumulate = accumulate + derivative*derivative
			learningRate = alpha/(delta+sqrt(accumulate))
			
			#print("learning rate is " + str(learningRate) )
			weight[i][0] = weight[i][0] - learningRate*derivative
			

		J_History[x][0] = computeErrorRate( trainData, yHead, weight )
		print("finish iteration " + str(x) + ", error is "+ str( J_History[x][0] ) )

	return weight, J_History

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



def OutputModel( weight, mean_r, std_r ):

	model = np.ones( shape = ( 3, featureNum+1 ) )
	model[ 0, : featureNum ] = mean_r
	model[ 1, : featureNum ] = std_r
	model[ 2, : ] = weight[ :, 0 ]

	fullModelName = modelName + '.npy' 
	np.save( fullModelName, model )

######Training		
dataList, label, count = loadData(fileName)
trainData = manageData( dataList, count )
trainData, mean_r, std_r = featureNormalization( trainData )
weight = zeros( shape = ( featureNum+1, 1 ) )
weight, J_History = gradientDescent( trainData, label, weight, count  )


######Output model
OutputModel( weight, mean_r, std_r )



