#include "PassDetails.h"

#include "toy/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

#define DEBUG_TYPE "toy-loop-unroll"

using namespace mlir;

namespace {
struct ToyLoopUnroll : public toy::ToyLoopUnrollBase<ToyLoopUnroll> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleOp.walk([&](affine::AffineForOp op) {
      (void)loopUnrollJamByFactor(op, 4);
    });
  }
};

} // end anonymous namespace

namespace mlir {
namespace toy {
std::unique_ptr<Pass> createToyLoopUnrollPass() {
  return std::make_unique<ToyLoopUnroll>();
}
} // namespace toy
} // namespace mlir
