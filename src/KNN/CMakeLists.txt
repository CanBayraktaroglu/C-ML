cmake_minimum_required(VERSION 3.10)

project(knn)

# Add the libraries
include_directories(./)

# Create an executable for test
add_executable(knn ./knn.c)

# Link the libraries
target_link_libraries(knn m)

# Link test against the libraries
target_include_directories(knn PUBLIC ./)
