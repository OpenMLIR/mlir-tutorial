add_executable(toy-ex1-io ex1-io.cpp)
target_link_libraries(
  toy-ex1-io
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)
