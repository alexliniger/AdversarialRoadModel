cmake_minimum_required(VERSION 3.13)
project(DiscKernelAlgo)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(External/Eigen/)
include_directories(External/Json/include)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(DiscKernelAlgo
        Model/model.cpp
        Algo/algo.cpp
        main.cpp
        types.h)
