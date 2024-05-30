#include "KNN.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

int main(){

	//Instantiate a KNN 
	KNN* knn = KNN_create();

	// Configs
	KNN_Config config;
	load_yaml_knn("../src/KNN/configs/configs.yaml", &config);

	//Run KNN
	KNN_run(knn, &config);	
	
	return 0;

}	
