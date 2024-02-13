cmake_minimum_required(VERSION 3.0)
project(Unfolder)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#SET(CMAKE_CXX_FLAGS "-g")
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
add_executable(Unfolder unfolding_pipeline.cpp new_skelet_conn.h correspondence.h alignment.h
	                    new_skelet_conn.cpp correspondence.cpp alignment.cpp
	                    utility.h utility.cpp st_fin_slice.h st_fin_slice.cpp)
target_link_libraries(Unfolder opencv_core opencv_imgproc opencv_ximgproc nlohmann_json)
