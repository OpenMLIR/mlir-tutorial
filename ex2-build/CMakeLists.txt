add_executable(toy-ex2-build ex2-build.cpp)
target_link_libraries(
  toy-ex2-build
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)
