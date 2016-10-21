import numpy as np
import csv
import math
from numpy import ones, zeros, mean, std
from math import sqrt

#parameter
iteration = 480
alpha = 1
delta = 0.0000001
featureNum = 57

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
				yHead.append( row[i] )
				continue
			data.append( row[i] )

		dataList.append(data)
		count = count+1

	yHead = np.matrix( yHead, dtype = np.float64 )
	dataList = np.matrix( dataList, dtype = np.float64 )
	print("count = " + str(count) )

	return dataList, yHead, count

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

def checkData( dataList, yHead ):

	for x in range( len(yHead) ):
		print("label = " + yHead[x] )
		print("data = " )
		print(dataList[x])

#add bias to dataList
def manageData( dataList, count ):

	trainData = ones( shape = ( count, featureNum+1) )
	trainData[ :, : featureNum ] = dataList[ :, : ]

	return trainData

def sigmoid( X ):

	den = 1.0 + np.exp( -1.0*X)
	d = 1.0/den

	return d

def computeCost( trainData, yHead, weight ):

	m = yHead.size
	#print("m = " + str(m) )
	prediction = sigmoid( trainData.dot(weight) )	

	loss = ((yHead*np.log(prediction) + (1-yHead)*np.log(1-prediction)).sum())/m

	return (-1)*loss

def computeErrorRate( trainData, yHead, weight ):

	m = yHead.size
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

		for i in range( len(weight) ):

			tmp = trainData[ :, i ]
			tmp.shape = ( count, 1 )

			derivative = ( ( ( prediction - yHead )*tmp ).sum() )/count
			accumulate = accumulate + derivative*derivative
			learningRate = alpha/(delta+sqrt(accumulate))

			#print("learning rate is " + str(learningRate) )
			weight[i][0] = weight[i][0] - learningRate*derivative

		J_History[x][0] = computeErrorRate( trainData, yHead, weight )
		print("finish iteration " + str(x) + ", error is "+ str( J_History[x][0] ) )

	return J_History


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
	f = open('logisticRegression.csv', 'w' )
	w = csv.writer(f)
	w.writerows(result)
	f.close()

		
dataList, yHead, count = loadData('spam_data/spam_train.csv')
#checkData(dataList, yHead )
trainData = manageData( dataList, count )
weight = ones( shape = ( featureNum+1, 1 ) )
J_History = gradientDescent( trainData, yHead, weight, count  )

#print(J_History)

testData, testDataCount = loadTestData('spam_data/spam_test.csv')
result = computeTestDataResult( testData, weight )
writeCSV(result)



