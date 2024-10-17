cmake_minimum_required(VERSION 3.10)

project(C_ML)
set(CMAKE_C_STANDARD 99)

# Add the libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML REQUIRED yaml-0.1)

include_directories(../src/utils/ ../src/DT ../src/regression/LR ../src/DeepLearning/ ${YAML_INCLUDE_DIRS})

# Create an executable for test
add_executable(knn ../src/KNN/knn.c)
add_executable(dt ../src/DT/dt.c)
add_executable(dl ../src/DeepLearning/dl.c)
add_executable(test ../src/DeepLearning/test.c)

# Link the libraries
target_link_libraries(knn m ${YAML_LIBRARIES})
target_link_libraries(dl m ${YAML_LIBRARIES})
target_link_libraries(dt m ${YAML_LIBRARIES})
target_link_libraries(test m ${YAML_LIBRARIES})

# Link test against the libraries
#target_include_directories(knn PUBLIC ./)
#target_include_directories(dt PUBLIC ./)
