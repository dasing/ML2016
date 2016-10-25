import numpy as np
import csv
import math
from numpy import ones, zeros, mean, std
from math import sqrt
import time

#parameter
iteration = 100000
alpha = 1
delta = 0.0000001
featureNum = 57
batchSize = 10

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

	trainData = trainData.transpose()

	mean_r = []
	std_r = []
	trainData_norm = trainData

	for x in range( featureNum ):
		m = mean( trainData[ x, : ] )
		s = std( trainData[ x, : ] )
		mean_r.append(m)
		std_r.append(s)
		trainData_norm[ x, : ] = ( trainData_norm[ x, : ] - m )

	return trainData_norm, mean_r, std_r

def percentage2n(eigVals,percentage):

    sortArray=np.sort(eigVals)   #升序  
    sortArray=sortArray[-1::-1]  #逆转，即降序  
    arraySum=sum(sortArray)  
    tmpSum=0  
    num=0  
    for i in sortArray:  
        tmpSum+=i  
        num+=1  
        if tmpSum>=arraySum*percentage:  
            return num  


def pca( trainData, mean_r, std_r, percentage ):


	covTrainData = np.cov( trainData, rowvar = 0 )
	eigvals, eigvectors = np.linalg.eig( np.mat( covTrainData) )
	n = percentage2n(eigvals,percentage) 
	eigValIndice = np.argsort( eigvals )
	n_eigValIndice = eigValIndice[-1:-(n+1):-1]
	n_eigVect = eigVects[:,n_eigValIndice]
	lowDDataMat = trainData*n_eigVect
	print( lowDDataMat )
    #reconMat
    #reconMat = ( lowDDataMat*n_eigVect.T ) + meanVal  #重构数据  

def testDataNormalization( testData,  mean_r, std_r ):

	for x in range( featureNum ):
		testData[ x, : ] = ( testData[ x, :] - mean_r[x] ) / std_r[x]

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
	f = open('logisticRegression.csv', 'w' )
	w = csv.writer(f)
	w.writerows(result)
	f.close()

######Training		
dataList, label, count = loadData('spam_data/spam_train.csv')
trainData = manageData( dataList, count )
trainData, mean_r, std_r = featureNormalization( trainData )

pca( trainData, mean_r, std_r, 0.99 )

# weight = zeros( shape = ( featureNum+1, 1 ) )
# weight, J_History = gradientDescent( trainData, label, weight, count  )


# ######Testing
# testData, testDataCount = loadTestData('spam_data/spam_test.csv')
# testData = testDataNormalization( testData, mean_r, std_r )
# result = computeTestDataResult( testData, weight )
# writeCSV(result)

