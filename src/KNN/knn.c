#include "KNN.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

int main(){

	//Instantiate a KNN 
	KNN* knn = KNN_create();

	// Configs
	float split_ratio = 0.6;
	unsigned char k = 3;

	//Run KNN
	KNN_run(knn, split_ratio, k);	
	
	return 0;

}	
