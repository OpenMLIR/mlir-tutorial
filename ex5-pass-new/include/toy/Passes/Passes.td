#ifndef TOY_PASSES
#define TOY_PASSES

include "mlir/Pass/PassBase.td"

def ToyLoopUnroll : Pass<"toy-loop-unroll", "mlir::ModuleOp"> {
  let summary = "Loop unroll";
  let constructor = "toy::createToyLoopUnrollPass()";
}

#endif // TOY_PASSES
