import csv
import numpy as np
from math import floor, sqrt
from numpy import ones, zeros, mean, std

#parameter
stride = 1 #sample rate 
datafeatureNum = 18
featureNum = 18 
iteration = 1000000
alpha = 1 #initialization of learning rate
delta = 0.0000001 #adagrad parameter, for nurmical stability

item = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR' ]
list = []


for x in range( 0, len(item) ):
	list.append([])

#read file and store it in list
def loadData(file_name):
	f = open( file_name, 'r', encoding = "ISO-8859-1" )  
	for row in csv.reader(f):  
	    #print(row)
	    try:
	    	idx = item.index(row[2])
	    	#print(row[2] + " found at " + str(idx) )

	    	for x in range( 3, 27):
	    		list[idx].append(row[x])
		
	    except ValueError:
	    	print(row[2] + " is not found")


def loadTestData( file_name, itemName ):

	testData = []
	f = open( file_name, 'r', encoding = "ISO-8859-1" )  
	data = []
	n = 0
	for row in csv.reader(f):
		#print(row)
		if n == 18:
			data.append(1.0) #add bias
			# print("add data to dataset")
			# print(data)
			testData.append(data)
			n = 0
			data = []
		
		for x in range( 2, 11 ):
			if row[x] == 'NR':
				data.append(0.0)
				continue
			if float(row[x]) < 0:
				if item[n] != 'AMB_TEMP':
					#compute value by neighbor
					leftNeighbor = 0
					rightNeighbor = 0
					leftNeighborValue = 0
					rightNeighborValue = 0
					
					
					if x > 2:
						leftNeighbor = x -1 
						leftNeighborValue =  float(row[leftNeighbor]) if float(row[leftNeighbor]) > 0 else 0
					if x < 10:
						rightNeighbor = x+1
						rightNeighborValue = float(row[rightNeighbor]) if float(row[rightNeighbor]) > 0 else 0

					leftNeighborWeight = leftNeighbor-1 if leftNeighbor-1 > 0 else 0
					rightNeighborWeight = rightNeighbor-1 if rightNeighbor-1 > 0 else 0

					# print("n = " + str(n))
					# print("leftNeighbor = " + str(leftNeighbor))
					# print("rightNeighbor = " + str(rightNeighbor))
					# print("leftNeighbor value = " + str(leftNeighborValue))
					# print("rightNeighbor value = " + str(rightNeighborValue) )
					value = ( leftNeighborValue*leftNeighborWeight + rightNeighborValue*rightNeighborWeight ) / ( leftNeighborWeight + rightNeighborWeight )
					#print("value = " + str(value))
                    #PM2.5 are all integer
					if n == 9:
						value = round(value)

					row[x] = str(value)
					data.append(value)
					continue
			#print(row[x])
			data.append(row[x])
			
		n = n +1
				
	data.append(1.0)
	testData.append(data)	

	# for x in testData:
	# 	print(len(x))
	# 	print(x)


	return np.matrix(testData, dtype = np.float64)


def genData(itemName):

	idx = item.index(itemName)
	data = []
	yHead = []
	count = 0
	dataPerMonth = 480

	for month in range(12):

		for i in range( 9, dataPerMonth, stride ):
			d = []
			ans = -1
			idx = dataPerMonth*month+i

			for featureIndex in range( 0, datafeatureNum ):

				flag = 0
				
				for x in range( 0, 10 ):

					if x == 9: #only pm2.5 shold add ans to data list
						if item[featureIndex] != 'PM2.5':
							continue
						else :
							if float(list[featureIndex][idx-(9-x)]) == float(-1):  # PM2.5 cannot be 1 
								flag = 1
								break
							else:
								ans = float(list[featureIndex][idx-(9-x)])
								continue
							
					if item[featureIndex] == 'RAINFALL':
						if list[featureIndex][idx-(9-x)] == 'NR':
							d.append(0.0)
							continue
					if float(list[featureIndex][idx-(9-x)]) < 0:
						if item[featureIndex] != 'AMB_TEMP': #other feature may not < 0
							flag = 1
							break
					
					d.append(list[featureIndex][idx-(9-x)])

				if flag == 1 :
					break

			
			if flag == 0 and ans >= 0:
				d.append(1.0)
				data.append(d)
				yHead.append(ans)
				count = count+1

	# print("yHead = " + str(len(yHead)))
	# print("count = " + str(count) )
	if len(yHead) != count:
		print("Error! yHead != count ")
		

	#change it from list to matrix
	it = np.matrix( data, dtype = np.float64 )

	return it, count, yHead

#check if data loaded is correct
def checkData( count, it, yHead ):

	for i in range(count):
		print("Ans: " + str(yHead[i][0]) )
		print(it[i])


def computeCost( it, yHead, theta ):
	m = yHead.size
	prediction = it.dot(theta)
	loss = prediction - yHead
	sqError = np.sum( loss**2 ) / (2*m)
	return sqError

def gradientDescent( it, yHead, iteration, alpha, theta ):

	m = len(yHead)
	J_History = zeros( shape = ( iteration, 1 ) )
	accumulate = 0

	for i in range( 0, iteration ):

		#compute prediction
		prediction = it.dot(theta)
		
		for x in range( len(theta) ):
			tmp = it[ :, x ]
			tmp.shape = ( m, 1 )
			
			#compute gradient
			derivative =  ( (( prediction - yHead )*tmp ).sum() )/m 
			accumulate = accumulate + derivative*derivative
			learningRate = alpha/( delta+sqrt(accumulate) ) 

			#update theta
			theta[x][0] = theta[x][0] - learningRate*derivative

		#compute cost 
		J_History[i][0] = computeCost( it, yHead, theta )
		
	return theta, J_History


def computeTestResult( testData, theta ):
	result = [['id', 'value']]
	prediction = testData.dot(theta)
	#print(prediction)
	for x in range( 0, len(prediction)):
		r = []
		id = "id_" + str(x)
		r.append(id)
		r.append( prediction.item(x,0) )
		#print(r)
		#print("id_" + str(x) + " , prediction = " + str(prediction[x][0]) )
		result.append(r)

	return result

def writeCSV( result ):
	f = open('kaggle_best.csv', 'w')
	w = csv.writer(f)
	w.writerows(result)
	f.close()

def changeDataType( it, count, yHead ):

	trainData = ones(shape = ( count, featureNum*9+1 ) )
	trainData[ :, : ] = it[ 0: count, : ]

	trainLabel = zeros( shape = (count, 1 ) )
	for x in range(count):
		trainLabel[ x, 0 ] = yHead[x]

	return trainData, trainLabel


def featureNormalization(it):

	mean_r = []
	std_r = []
	it_norm = it

	for x in range( 0, featureNum*9, 9 ):
		m = mean(it[ : , x : x+9 ])
		s = std(it[ :, x: x+9 ])
		mean_r.append(m)
		std_r.append(s)
		it_norm[ : , x : x+9 ] = ( it_norm[ :, x : x+9 ] - m ) / s

	return it_norm, mean_r, std_r


def testDataNormalization( testData, mean_r, std_r ):

	for x in range( 0, featureNum*9, 9 ):
		testData[ :, x : x+9 ] = ( testData[ :, x : x+9 ] - mean_r[floor(x/9)] ) /std_r[floor(x/9)]

	return testData 

######training

loadData('data/train.csv')
it, count, yHead = genData('PM2.5')
it, trainLabel = changeDataType( it, count, yHead )
theta = zeros( shape = ( 9*featureNum+1, 1 ) )
#checkData(count, it, trainLabel)

it, mean_r, std_r = featureNormalization( it )
#print(it)

theta, J_History = gradientDescent( it, trainLabel, iteration, alpha, theta )
print(theta)
for x in J_History:
	print(x)

# # #######testing

testData = loadTestData('data/test_X.csv', 'PM2.5')
testData = testDataNormalization( testData, mean_r, std_r )
# n = 0
# for row in testData:
#     print("data " + str(n) )
#     print(row)
#     n = n+1
result = computeTestResult( testData, theta )
writeCSV( result )



