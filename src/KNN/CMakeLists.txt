cmake_minimum_required(VERSION 3.10)

project(knn)

# Add the libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML REQUIRED yaml-0.1)
include_directories(./ ${YAML_INCLUDE_DIRS})
# Create an executable for test
add_executable(knn ./knn.c)

# Link the libraries
target_link_libraries(knn m ${YAML_LIBRARIES})

# Link test against the libraries
target_include_directories(knn PUBLIC ./)
