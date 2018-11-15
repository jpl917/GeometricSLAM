cmake_minimum_required(VERSION 2.8)
project(lsd-1.5)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

find_package(OpenCV REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED 
  lsd-1.5/lsd.c)

add_executable( lsd_sample lsd-1.5/lsd_call_example.c )
target_link_libraries(lsd_sample ${PROJECT_NAME})

