cmake_minimum_required(VERSION 2.6)
project(mih)

# Force CMake to prefer static libraries.
set(BUILD_SHARED_LIBS OFF)
# Help CMake find static libraries.
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib")

find_library(HDF5_LIB hdf5)

set(CMAKE_CXX_FLAGS "-DSINGLE_CORE -O2 -fopenmp -DNDEBUG")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(/usr/include/hdf5/serial)

add_executable(mih
    src/array32.cpp
    src/mihasher.cpp
    src/sparse_hashtable.cpp
    src/bucket_group.cpp
    interface/mih_interface.cpp
    src/reorder.cpp
    interface/loadVar.cpp
    interface/saveRes.cpp
)

# Link HDF5 and its static dependencies.
target_link_libraries(mih ${HDF5_LIB} z sz curl crypto ssl)

project(linscan)
add_executable(linscan
    interface/linscan_interface.cpp
    src/linscan.cpp
    interface/loadVar.cpp
    interface/saveRes.cpp
)
target_link_libraries(linscan ${HDF5_LIB} z sz curl crypto ssl)
