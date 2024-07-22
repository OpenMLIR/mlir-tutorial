#include "PassDetails.h"

#include "mlir/IR/Dominance.h"
#include "toy/Passes/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#define DEBUG_TYPE "toy-iter-args-opt"

using namespace mlir;

bool isInCurrentAffineFor(Operation *op, affine::AffineForOp forOp) {
  auto *parentOp = op->getParentOp();
  auto maybeParentFor = dyn_cast_or_null<affine::AffineForOp>(parentOp);
  return maybeParentFor && maybeParentFor == forOp;
}

bool areInSameAffineFor(affine::AffineLoadOp load, affine::AffineStoreOp store,
                        affine::AffineForOp forOp) {
  return isInCurrentAffineFor(load, forOp) &&
         isInCurrentAffineFor(store, forOp);
}

bool IsParentOp(Operation *op, Operation *maybeParentOp) {
  auto iterOp = op;
  // 不断迭代op的getParentOp
  while (auto parentOp = iterOp->getParentOp()) {
    if (parentOp == maybeParentOp)
      return true;
    iterOp = parentOp;
  }
  return false;
}

bool isDominance(Operation *maybeDominateOp, Operation *op) {
  DominanceInfo dom(maybeDominateOp);
  // 利用 支配关系判断，执行op前maybeDominateOp一定被执行过，
  // properly会判断不在同一个块
  return dom.properlyDominates(maybeDominateOp, op);
}

bool checkloadCanPromote(affine::AffineForOp forOp, affine::AffineLoadOp load,
                         Operation *&storeInfor) {
  Value memref = load.getMemRef();
  auto mt = memref.getType().cast<MemRefType>();
  // 只针对memref为1个元素
  if (mt.getShape().size()) {
    return false;
  }
  bool storeInforFlag = false;
  // 获取 def-use chains
  for (auto *user : memref.getUsers()) {
    if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
      // for循环内的同级store
      if (areInSameAffineFor(load, store, forOp)) {
        if (storeInforFlag) {
          // 仅允许出现一次store
          return false;
        }
        storeInforFlag = true;
        // 检查到达 store 都必须经过 这次load，且不在一个block
        if (!isDominance(load, store)) {
          return false;
        }
        storeInfor = store;
      } else if (IsParentOp(store, forOp)) {
        // for region 内还有其他store，不优化
        return false;
      }
    } else if (auto otherLoad = dyn_cast<affine::AffineLoadOp>(user)) {
      if (load != otherLoad && IsParentOp(otherLoad, forOp)) {
        // for region 内有其他 load，不优化
        return false;
      }
    }
  }
  // debug 时打印优化的memref
  LLVM_DEBUG(llvm::dbgs() << " Can promte to iter_args: " << memref << "\n");

  return true;
}

// 从 mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:1104 复制并做了修改
affine::AffineForOp replaceForOpWithNewSignature(OpBuilder &builder,
                                                 affine::AffineForOp loop,
                                                 Value iterValue) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(loop);

  builder.setInsertionPoint(loop);
  SmallVector<Value, 4> operands;
  llvm::append_range(operands, loop.getRegionIterArgs());
  operands.push_back(iterValue);

  auto newLoop = builder.create<affine::AffineForOp>(
      loop.getLoc(), loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
      loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
      loop.getStepAsInt(), operands);
  newLoop.getBody()->erase();

  newLoop.getRegion().getBlocks().splice(
      newLoop.getRegion().getBlocks().begin(), loop.getRegion().getBlocks());

  newLoop.getBody()->addArgument(iterValue.getType(), iterValue.getLoc());

  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  LLVM_DEBUG(llvm::dbgs() << "newLoop now: " << newLoop << "\n");
  LLVM_DEBUG(llvm::dbgs() << "stripped affine.for: " << loop << "\n");
  LLVM_DEBUG(llvm::dbgs() << "erase: " << loop);

  loop->erase();
  return newLoop;
}

void replaceWithNewFor(affine::AffineForOp forOp, Operation *load,
                       Operation *store) {
  OpBuilder builder(forOp);
  builder.setInsertionPoint(forOp);
  auto movedLoad = builder.clone(*load);
  auto newLoop =
      replaceForOpWithNewSignature(builder, forOp, movedLoad->getResult(0));

  // update yieldOp
  auto forYieldOp =
      cast<affine::AffineYieldOp>(newLoop.getBody()->getTerminator());
  forYieldOp->insertOperands(forYieldOp.getNumOperands(), store->getOperand(0));

  // 重写AffineStoreOp
  builder.setInsertionPointAfter(newLoop);
  auto affineStore = cast<affine::AffineStoreOp>(store);

  // store 循环的返回值
  builder.create<affine::AffineStoreOp>(
      newLoop.getLoc(), newLoop.getResults()[newLoop.getNumResults() - 1],
      affineStore.getMemRef(), affineStore.getAffineMap(),
      affineStore.getMapOperands());

  // 修改load的值为for的最后一个iter_args
  load->getResult(0).replaceAllUsesWith(
      newLoop.getBody()->getArgument(newLoop.getBody()->getNumArguments() - 1));
  // 删除多余的op
  load->erase();
  store->erase();
}

// affine.for %arg1 = 0 to 10 {
//   %1 = affine.load %arg0[%arg1]
//   %2 = affine.load %alloca[] : memref<type>
//   %3 = arith.addf %2, %1
//   affine.store %3, %alloca[] : memref<type>
// }
//
// becomes
//
// %0 = affine.load %alloca[] : memref<type>
// %1 = affine.for %arg1 = 0 to 10 iter_args(%arg2 = %0) -> (type) {
//   %3 = affine.load %arg0[%arg1]
//   %4 = arith.addf %arg2, %3
//   affine.yield %4
// }
// affine.store %1, %alloca[] : memref<type>

namespace {
struct ToyMem2IterArgs : public toy::ToyMem2IterArgsBase<ToyMem2IterArgs> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    // 从for循环入手
    moduleOp.walk([&](affine::AffineForOp forOp) {
      bool isCanPromote = false;
      Operation *canPromotetLoadOp, *canPromoteStoreOp;
      forOp->walk([&](affine::AffineLoadOp load) {
        if (checkloadCanPromote(forOp, load, canPromoteStoreOp)) {
          isCanPromote = true;
          canPromotetLoadOp = load;
          // 可以优化，结束walk
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (isCanPromote) {
        replaceWithNewFor(forOp, canPromotetLoadOp, canPromoteStoreOp);
      }
    });
  }
};

} // end anonymous namespace

namespace mlir {
namespace toy {
std::unique_ptr<Pass> createToyMem2IterArgsPass() {
  return std::make_unique<ToyMem2IterArgs>();
}
} // namespace toy
} // namespace mlir
