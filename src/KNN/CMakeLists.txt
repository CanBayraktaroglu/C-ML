cmake_minimum_required(VERSION 3.10)

project(knn)

# Add the libraries
include_directories(./)

# Create an executable for test
add_executable(main ./main.c)

# Link the libraries
target_link_libraries(main m)

# Link test against the libraries
target_include_directories(main PUBLIC ./)
