#include "dataset.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

int main(){

	//Instantiate a decision tree

	// Configs
	DT_Config config;
	load_yaml_knn("../src/KNN/configs/configs.yaml", &config);

	//Run KNN
	DT_run(dt, &config);	
	
	return 0;

}	