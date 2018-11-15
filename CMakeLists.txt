cmake_minimum_required( VERSION 2.8 )
project(line_slam )

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_BUILD_TYPE "Release")
set( CMAKE_CXX_FLAGS "-std=c++11")
#set( CMAKE_CXX_FLAGS "-O3" )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)# COMPONENTS common io)
find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CSparse REQUIRED)
find_package(Pangolin REQUIRED)
Find_Package(Cholmod REQUIRED)


include_directories(/usr/include/python2.7/)  
link_directories(/usr/lib/python2.7/config-x86_64-linux-gnu/)  

include_directories("peac")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${G2O_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${CSPARSE_INCLUDE_DIR}
    ${CHOLMOD_INCLUDE_DIR}
)

#message("find opencv in " ${OpenCV_INCLUDE_DIRS})
#message("find pcl in " ${PCL_INCLUDE_DIRS})
#message("find project in " ${PROJECT_SOURCE_DIR})
#message("find g2o in " ${G2O_INCLUDE_DIR})
#message("find eigen in " ${EIGEN3_INCLUDE_DIR})
#message("find csparse in " ${CSPARSE_INCLUDE_DIR})
#message("find cholmod in " ${CHOLMOD_INCLUDE_DIR})

link_directories(${PCL_LIBRARY_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/levmar-2.6
)

SET(Line_LIBS levmar )
SET(Line_LIBS ${Line_LIBS} armadillo) 

add_definitions(${PCL_DEFINITIONS})

add_library(${PROJECT_NAME} SHARED 
    lsd.cpp
    utils.cpp
    frame.cpp
    PnPsolver.cpp
    edge_se3_lineendpts.cpp
    vertex_lineendpts.cpp
    motion.cpp
    Viewer.cpp
    python.cpp
)


SET(G2O_LIBS g2o_cli g2o_core g2o_csparse_extension g2o_ext_freeglut_minimal g2o_incremental g2o_interactive g2o_interface 
g2o_opengl_helper g2o_parser g2o_simulator g2o_solver_cholmod g2o_solver_csparse g2o_solver_dense g2o_solver_eigen g2o_solver_pcg 
g2o_solver_slam2d_linear g2o_solver_structure_only g2o_stuff g2o_types_data g2o_types_icp g2o_types_sba g2o_types_sclam2d g2o_types_sim3
g2o_types_slam2d_addons g2o_types_slam2d g2o_types_slam3d_addons g2o_types_slam3d g2o_viewer cxsparse )

target_link_libraries(${PROJECT_NAME}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PCL_LIBRARIES}
${OpenCV_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/EDLines/libEDLines.a
#${PROJECT_SOURCE_DIR}/Thirdparty/levmar-2.6/liblevmar.a
/usr/lib/x86_64-linux-gnu/libcholmod.so.2.1.2
${G2O_LIBS}
)




#build samples 
#add_executable( slam main.cpp )
#target_link_libraries( slam ${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${G2O_LIBS})


add_executable( naive_slam slam.cpp )
target_link_libraries( naive_slam ${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${G2O_LIBS} ${Line_LIBS})

add_executable( test test.cpp )
target_link_libraries( test ${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${G2O_LIBS} ${Line_LIBS})

#add_executable( 1 1.cpp )
#target_link_libraries( 1 ${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${G2O_LIBS} ${Line_LIBS})

add_executable( lineslam lineslam.cpp )
target_link_libraries( lineslam ${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${G2O_LIBS} ${Line_LIBS})

add_executable(python python.cpp)  
target_link_libraries(python -lpython2.7 ${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${G2O_LIBS} ${Line_LIBS} ${OpenCV_LIBS})  

add_executable(peac peac.cpp)  
target_link_libraries(peac ${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${G2O_LIBS} ${Line_LIBS})  
