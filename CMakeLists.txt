cmake_minimum_required (VERSION 3.18)

list(APPEND CMAKE_PREFIX_PATH "/install/lib/cmake/alpaka/")
set(_TARGET_NAME myProject)
project(${_TARGET_NAME})
include_directories(${myProject_SOURCE_DIR}/"headers")
include_directories(${myProject_SOURCE_DIR}/"geometry")
find_package(alpaka REQUIRED)


alpaka_add_executable(${_TARGET_NAME} main.cpp)
target_link_libraries(
  ${_TARGET_NAME}
  PUBLIC alpaka::alpaka)
