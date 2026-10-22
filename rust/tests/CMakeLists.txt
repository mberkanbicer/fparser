cmake_minimum_required(VERSION 3.1...3.10)
project(fparser_rust_reference CXX)

get_filename_component(FPARSER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
    fparser-reference
    "${FPARSER_ROOT}/fparser.cc"
    "${FPARSER_ROOT}/fpoptimizer.cc"
    original_reference.cpp
)
target_include_directories(fparser-reference PRIVATE "${FPARSER_ROOT}")
