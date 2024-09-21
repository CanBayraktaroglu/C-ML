#ifndef LOGGER_H_
#define LOGGER_H_

#include "vector.h"
#include "math.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"
#include <string.h>

typedef enum{
    DEFAULT,
    DEBUG,
    WARNING,
    ERROR
}Logger_Level;

 typedef struct{
    void (*log)(char* string);
    void (*debug)(char* string);
    void (*warn)(char* string);
    void (*err)(char* string);        
 }Logger;

void logg(string){
    printf(string);
    printf("\n");    
};

void err(const char* string){
    printf("---------- ERROR ----------\n");
    printf(string);
    printf("\n");
};

void debug(const char* string){
    printf("---------- DEBUG ----------\n");
    printf(string);
    printf("\n");
};

void warn(const char* string){
    printf("---------- WARNING ----------\n");
    printf(string);
    printf("\n");
};

Logger init_Logger(){
    Logger logger;
    logger.log = logg;
    logger.debug = debug;
    logger.warn = warn;
    logger.err = err;

    return logger;
};

#endif LOGGER_H_

