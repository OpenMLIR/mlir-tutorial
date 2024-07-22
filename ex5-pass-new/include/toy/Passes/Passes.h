#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>

namespace mlir {
namespace toy {
std::unique_ptr<Pass> createToyLoopUnrollPass();
std::unique_ptr<Pass> createToyMem2IterArgsPass();
} // namespace toy
} // namespace mlir

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "toy/Passes/Passes.h.inc"

} // end namespace mlir

#endif // TOY_PASSES_H
