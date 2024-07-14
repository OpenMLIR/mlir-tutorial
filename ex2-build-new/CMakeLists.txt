cmake_minimum_required(VERSION 3.13.4)

project(toy-build LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

add_executable(toy-build toy-build.cpp)
target_link_libraries(
  toy-build
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAffineUtils
)