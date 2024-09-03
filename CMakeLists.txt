cmake_minimum_required(VERSION 3.13)
project(stratosml C CXX)

find_package(Armadillo REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR})
# include_directories(/usr/include/armadillo)
#TODO link_directories()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

## recurse
add_subdirectory(src/stratosml) # mlpack

target_link_libraries(stratosml PUBLIC armadillo)

set_target_properties(stratosml PROPERTIES LINKER_LANGUAGE CXX)