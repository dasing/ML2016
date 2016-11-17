execution:
./train.sh $1 $2
/* 
	$1-> dir path constains train and test data, shold add '/' at the end of the path, ex: 'data/'
	$2-> output modelName
*/

./test.sh $1 $2 $3
/*
	$1-> dir path constains train and test data, shold add '/' at the end of the path, ex: 'data/'
	$2-> output model, should not contain 'fileName Extension'
	$3-> output csv fileName
*/