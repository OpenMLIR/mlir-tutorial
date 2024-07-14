#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

int main(int argc, char **argv) {
  mlir::MLIRContext ctx;

  // Context 加载FuncDialect，MemRefDialect, AffineDialect 和 ArithDialect
  ctx.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect, mlir::arith::ArithDialect>();

  // 创建 OpBuilder
  mlir::OpBuilder builder(&ctx);

  // 创建IR的根，ModuleOp
  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

  // 设置插入点
  builder.setInsertionPointToEnd(module.getBody());

  // 创建 函数
  auto f32 = builder.getF32Type();
  // 创建 长度为 10 的数组
  auto memref = mlir::MemRefType::get({10}, f32);
  // 创建 func，函数名为ArraySum，输入是刚创建的数组，输出是f32
  auto func = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), "ArraySum",
      builder.getFunctionType({memref}, {f32}));

  // 设置插入点，插入到func所建的block后面
  builder.setInsertionPointToEnd(func.addEntryBlock());
  // 创建浮点类型的1.0
  mlir::Value constantFloatZeroVal = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getF32FloatAttr(0.0));
  // 存储sum的memref
  auto sumMemref = mlir::MemRefType::get({}, f32);
  // 创建sum的AllocaOp
  mlir::Value sumMemrefVal = builder.create<mlir::memref::AllocaOp>(
      builder.getUnknownLoc(), sumMemref);
  // 创建访问sum的空AffineMap
  auto sumMap = builder.getEmptyAffineMap();
  // 使用 store 初始化为0
  builder.create<mlir::affine::AffineStoreOp>(
      builder.getUnknownLoc(), constantFloatZeroVal, sumMemrefVal, sumMap,
      mlir::ValueRange());

  // 创建 lower bound AffineMap
  auto lbMap = mlir::AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                                    builder.getAffineConstantExpr(0),
                                    builder.getContext());
  // 创建 upper bound AffineMap
  auto ubMap = mlir::AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                                    builder.getAffineConstantExpr(10),
                                    builder.getContext());
  // 创建循环
  auto affineForOp = builder.create<mlir::affine::AffineForOp>(
      builder.getUnknownLoc(), mlir::ValueRange(), lbMap, mlir::ValueRange(),
      ubMap, 1);
  auto savedIP = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(affineForOp.getBody());

  // 为 %arg0[%arg1] 创建 AffineMap，表达式为 (d0) -> (d0)。即输入d0，结果为d0
  auto forLoadMap = mlir::AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/0, builder.getAffineDimExpr(0),
      builder.getContext());
  // Load %arg0[%arg1]
  mlir::Value affineLoad = builder.create<mlir::affine::AffineLoadOp>(
      builder.getUnknownLoc(), func.getArgument(0), forLoadMap,
      mlir::ValueRange(affineForOp.getBody()->getArgument(0)));
  // Load %alloca[]
  mlir::Value sumInforLoad = builder.create<mlir::affine::AffineLoadOp>(
      builder.getUnknownLoc(), sumMemrefVal, sumMap, mlir::ValueRange());
  // %alloca[] + %arg0[%arg1]
  mlir::Value add = builder.create<mlir::arith::AddFOp>(
      builder.getUnknownLoc(), sumInforLoad, affineLoad);
  // 保存到 %alloca[]
  builder.create<mlir::affine::AffineStoreOp>(
      builder.getUnknownLoc(), add, sumMemrefVal, sumMap, mlir::ValueRange());

  // 恢复InsertionPoint
  builder.restoreInsertionPoint(savedIP);
  // Load %alloca[]
  mlir::Value sumLoadVal = builder.create<mlir::affine::AffineLoadOp>(
      builder.getUnknownLoc(), sumMemrefVal, sumMap, mlir::ValueRange());
  // 创建以%alloca[]的结果的返回Op
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), sumLoadVal);
  module->print(llvm::outs());
  return 0;
}