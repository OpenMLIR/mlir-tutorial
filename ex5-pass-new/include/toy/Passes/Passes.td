#ifndef TOY_PASSES
#define TOY_PASSES

include "mlir/Pass/PassBase.td"

def ToyLoopUnroll : Pass<"toy-loop-unroll", "mlir::ModuleOp"> {
  let summary = "Loop unroll";
  let constructor = "toy::createToyLoopUnrollPass()";
}

def ToyMem2IterArgs : Pass<"toy-mem-to-iter-args", "mlir::ModuleOp"> {
  let summary = "memref load/store to affine.for iter_args";
  let constructor = "toy::createToyMem2IterArgsPass()";
}

#endif // TOY_PASSES
