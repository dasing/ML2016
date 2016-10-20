import numpy as np
import csv
import math
from numpy import ones, zeros, mean, std

#parameter
iteration = 100
alpha = 0.000001
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

	# length = (X.shape)[0]
	# for i in range( length ):
	# 	#print("i = " + str(i) +" , origin value = " + str( X[ i, 0 ]) )
	# 	X[ i, 0 ] = 1 / ( 1 + math.exp( -X[ i, 0 ] ) )
	# 	#print("afterValue = " + str( X[ i, 0]))
	# return X
	den = 1.0 + np.exp( -1.0*X)
	d = 1.0/den

	return d

def computeCost( trainData, yHead, weight ):

	m = yHead.size
	#print("m = " + str(m) )
	prediction = sigmoid( trainData.dot(weight) )		
	loss = ((yHead*np.log(prediction) + (1-yHead)*np.log(1-prediction)).sum())/m

	return (-1)*loss


def gradientDescent( trainData, yHead, weight, count ):

	J_History = zeros( shape = ( iteration, 1 ) )

	for x in range( 0, iteration ):

		prediction =  sigmoid( trainData.dot(weight) )

		for i in range( len(weight) ):

			tmp = trainData[ :, i ]
			tmp.shape = ( count, 1 )

			derivative = ( ( ( prediction - yHead )*tmp ).sum() )/count
			weight[i][0] = weight[i][0] - alpha*derivative

		J_History[x][0] = computeCost( trainData, yHead, weight )
		print("finish iteration " + str(x) + ", error is "+ str( J_History[x][0] ) )


		
dataList, yHead, count = loadData('spam_data/spam_train.csv')
#checkData(dataList, yHead )
trainData = manageData( dataList, count )
weight = ones( shape = ( featureNum+1, 1 ) )
J_History = gradientDescent( trainData, yHead, weight, count  )
print(J_History)
