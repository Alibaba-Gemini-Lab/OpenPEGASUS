cmake_minimum_required(VERSION 3.10)
project(pegasus)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(PEGASUS_THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/thirdparty)

add_compile_options(-fPIC)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(thirdparty/Gemini-SEAL/)
include_directories(thirdparty/Gemini-SEAL/native/src/)
include_directories(thirdparty/rapidjson/include)
include_directories(thirdparty/)

set(PEGASUS_USE_EIGEN_STR "Build Eigen for Pegasus")
option(PEGASUS_USE_EIGEN ${PEGASUS_USE_EIGEN_STR} ON)

# Download and configure
if (PEGASUS_USE_EIGEN)
  message(STATUS "Setting up Eigen ...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    OUTPUT_QUIET
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${PEGASUS_THIRDPARTY_DIR}/eigen)
  if(result)
    message(WARNING "Failed to download Eigen (${result}); disabling `PEGASUS_USE_EIGEN`")
  endif()
  set(INSTALL_EIGEN OFF CACHE BOOL "" FORCE)
  mark_as_advanced(INSTALL_EIGEN)
  endif()

if (PEGASUS_USE_EIGEN)
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    OUTPUT_QUIET
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${PEGASUS_THIRDPARTY_DIR}/eigen)
  if(result)
    message(WARNING "Failed to build Eigen (${result}); disabling `PEGASUS_USE_EIGEN`")
  endif()

  add_subdirectory(${PEGASUS_THIRDPARTY_DIR}/eigen/src EXCLUDE_FROM_ALL)
  set(EIGEN_INCLUDE_DIR ${PEGASUS_THIRDPARTY_DIR}/eigen/src)
  include_directories(${EIGEN_INCLUDE_DIR})
endif()

find_package(GMP)
if(GMP_FOUND)
add_definitions(-DGEMINI_USE_GMP)
include_directories(${GMP_INCLUDES} /usr/local/include)
endif(GMP_FOUND)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

add_library(pegasus OBJECT)
add_subdirectory(pegasus)

if (GMP_FOUND)
target_link_libraries(pegasus ${GMP_LIBRARIES})
endif(GMP_FOUND)
target_link_libraries(pegasus seal)

add_subdirectory(examples)
