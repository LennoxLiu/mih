cmake_minimum_required(VERSION 2.6)
project (mih)

find_library (HDF5_LIB hdf5)
#find_library(HDF5_LIB hdf5 PATHS /usr/lib/x86_64-linux-gnu/hdf5/serial NO_DEFAULT_PATH)

set (CMAKE_CXX_FLAGS "-DSINGLE_CORE -O2 -fopenmp -DNDEBUG")
# -Wall -march=native -mtune=native")

include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories(/usr/include/hdf5/serial)

add_executable (mih src/array32.cpp src/mihasher.cpp src/sparse_hashtable.cpp src/bucket_group.cpp interface/mih_interface.cpp src/reorder.cpp interface/loadVar.cpp interface/saveRes.cpp)
target_link_libraries (mih ${HDF5_LIB})

project (linscan)
add_executable (linscan interface/linscan_interface.cpp src/linscan.cpp interface/loadVar.cpp interface/saveRes.cpp)
target_link_libraries (linscan ${HDF5_LIB})
