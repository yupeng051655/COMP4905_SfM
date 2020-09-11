cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
project( structure_from_motion )

find_package( OpenCV 3.4.10 REQUIRED )
#find_package( PCL REQUIRED )
#include_directories( ${PCL_INCLUDE_DIRS} )
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

add_executable( structure_from_motion structure_from_motion.cpp )

#target_link_libraries(structure_from_motion ${PCL_LIBRARIES})
target_link_libraries(structure_from_motion ${OpenCV_LIBS} )




