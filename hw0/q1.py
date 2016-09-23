import numpy as np
import sys

#get sys argument
fileName = sys.argv[2]
columnNumber = int(sys.argv[1])

#readFile
f = open( fileName , 'r', encoding='UTF-8' )

list = []

while True:
	i = f.readline()
	if i == '' : break
	stringList = i.split()
	list.append( float(stringList[columnNumber]) )	
	# message = 'add ' + stringList[columnNumber] + ' to list '
	# print(message)

list.sort()
# print(list)
	
f.close()

#write file
i = 0
f = open( 'ans1.txt', 'w', encoding='UTF-8' )
for x in list:
	f.write( str(x) )
	if i != len(list)-1 : f.write(',')
	i+=1
	
f.close() 