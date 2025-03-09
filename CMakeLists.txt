cmake_minimum_required(VERSION 3.10)

project(C_ML)
set(CMAKE_C_STANDARD 99)

# Add the libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML REQUIRED yaml-0.1)

# Set include directories with absolute paths
include_directories(
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/DT
    ${CMAKE_SOURCE_DIR}/src/regression/LR
    ${CMAKE_SOURCE_DIR}/src/DeepLearning
    ${YAML_INCLUDE_DIRS}
)

# Create an executable for test
add_executable(knn src/KNN/knn.c)
add_executable(dt src/DT/dt.c)
add_executable(dl src/DeepLearning/dl.c)
add_executable(autodiff_test src/DeepLearning/tests/autodiff_compute_test.c)
add_executable(feed_forward_test src/DeepLearning/tests/feed_forward_test.c)

# Link the libraries
target_link_libraries(knn m ${YAML_LIBRARIES})
target_link_libraries(dl m ${YAML_LIBRARIES})
target_link_libraries(dt m ${YAML_LIBRARIES})
target_link_libraries(autodiff_test m ${YAML_LIBRARIES})
target_link_libraries(feed_forward_test m ${YAML_LIBRARIES})
