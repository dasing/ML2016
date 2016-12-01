import nltk
import re
import numpy as np
import csv
import sys
import random
from nltk.corpus import stopwords #Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

inputFilePath = sys.argv[1]
outputFileName = sys.argv[2]

titlePath = inputFilePath + 'title_StackOverflow.txt'
testDataPath = inputFilePath + 'check_index.csv'

#parameter
REMOVE_NON_LETTERS = True
REMOVE_STOP_WORDS = True
TITELSIZE = 20000
TAG_NUMBER = 20
DIMENSION = 500


def loadData():

	words = []
	fo = open( titlePath, "r" )
	for line in fo:
		title = ""

		
		letters_only = re.sub( "[^a-zA-Z]", " ", line )
		letters_only = letters_only.lower()
		title = letters_only


		if REMOVE_STOP_WORDS == True:
			letters_only_array = letters_only.split()
			#  In Python, searching a set is much faster than searching a list, so convert the stop words to a set
			stops = set( stopwords.words("english") )
			meaningful_words = [ w for w in letters_only_array if not w in stops ]

			# Join the words back into one string separated by space and return the result.
			wordString = " ".join( meaningful_words )
			title = wordString

		
		words.append( title )

	#print( words )

	return words

def BoW( words ):
	# Initialize the "CountVectorizer" object, which is scikit-learn bag of words tool.  
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = DIMENSION )
	train_data_features = vectorizer.fit_transform( words )
	train_data_features = train_data_features.toarray()

	# print ( train_data_features.shape )

	# # Take a look at the words in the vocabulary
	# vocab = vectorizer.get_feature_names()
	# print ( vocab )

	# # Sum up the counts of each vocabulary word
	# dist = np.sum(train_data_features, axis=0)

	# # For each, print the vocabulary word and the number of times it 
	# # appears in the training set
	# for tag, count in zip(vocab, dist):
	#     print (count, tag)


	return train_data_features

def findKinit( features ):

	init = np.zeros( shape = ( 21, DIMENSION ) )
	init[0] = features[0]
	count = 1

	while count < 20:

		x = random.randint( 0, TITELSIZE-1 )
		skipFlag = 1

		for feature in features[x]:
			if feature != 0:
				skipFlag = 0

		if skipFlag == 1:
			continue #all dimension are zeros, so discard this features

		for i in range( count ):
			if sum( p*q for p, q in zip( features[x], init[i] ) ) != 0 :
				skipFlag = 1
				break 
		
		if skipFlag == 0:
			#print("push " + str(x) + " to init")
			init[count] = features[x]
			count += 1

	extraInit = [ 0 ] * DIMENSION
	init[count] = extraInit

	return init

def KMeansAlgorithm( features ):

	#modelList = [ ]
	modelNumber = 15
	predict = np.zeros( shape = ( modelNumber, TITELSIZE ) )

	for i in range( modelNumber ):

		Kinit = findKinit( features )

		# for x in Kinit:
		# 	print( x ) 

		estimators = KMeans( init = Kinit, n_clusters = TAG_NUMBER +1 , n_init = 1 )
		model = estimators.fit( features )
		predict[i] = np.matrix( model.labels_ )

		#modelList.append( model )

	return predict

def EvaluateTestData( predict ):

	result = [['ID', 'Ans']]
	f = open( testDataPath, 'r' )
	
	n = 0
	for row in csv.reader(f):

		if n == 0 :
			n += 1
			continue
		#print(row)

		r = []
		for x in range(3):
			id = int(row[0])
			x_id = int(row[1])
			y_id = int(row[2])

			sameNum = 0
			diffNum = 0

			for m in range( predict.shape[0] ):
				x_predict = predict[m][ x_id ]
				y_predict = predict[m][ y_id ]

				if x_predict == 20 or y_predict == 20:
					diffNum += 1
				else:
					if x_predict == y_predict:
						sameNum += 1
					else:
						diffNum += 1

		r.append(id)
		if sameNum > diffNum:
			r.append(1)
		else:
			r.append(0)
		

		n += 1
		result.append(r)

	return result


def writeCSV( result ):

	fo = open( outputFileName, 'w' )
	w = csv.writer(fo)
	w.writerows(result)
	fo.close()

nltk.download("stopwords")

words = loadData()
features = BoW( words )
predict = KMeansAlgorithm( features )
result = EvaluateTestData( predict )
writeCSV( result )
