import numpy as np
import csv
import math
from numpy import ones, zeros, mean, std

#parameter
iteration = 10
alpha = 0.001
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

	dataList = np.matrix( dataList, dtype = np.float64 )

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

def sigmoid( x ):
	return( 1 / ( 1 + math.exp(-x) ))

def gradientDescent( trainData, yHead, weight ):

	for x in range( 0, iteration ):
		prediction =  sigmoid( trainData.dot(weight) )

		print( prediction )


dataList, yHead, count = loadData('spam_data/spam_train.csv')
#checkData(dataList, yHead )
trainData = manageData( dataList, count )
weight = ones( shape = ( featureNum+1, 1 ) )
gradientDescent( trainData, yHead, weight )
