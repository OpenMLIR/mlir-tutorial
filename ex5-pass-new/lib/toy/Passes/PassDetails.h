#ifndef DIALECT_TOY_TRANSFORMS_PASSDETAILS_H
#define DIALECT_TOY_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"
#include "toy/Passes/Passes.h"

namespace mlir {
namespace toy {

#define GEN_PASS_CLASSES
#include "toy/Passes/Passes.h.inc"

} // namespace toy
} // namespace mlir

#endif // DIALECT_TOY_TRANSFORMS_PASSDETAILS_H
