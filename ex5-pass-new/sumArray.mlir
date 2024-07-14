module attributes {} {
  func.func @ArraySum(%arg0: memref<10xf32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %cst, %alloca[] : memref<f32>
    affine.for %arg1 = 0 to 10 {
      %1 = affine.load %arg0[%arg1] : memref<10xf32>
      %2 = affine.load %alloca[] : memref<f32>
      %3 = arith.addf %2, %1 : f32
      affine.store %3, %alloca[] : memref<f32>
    }
    %0 = affine.load %alloca[] : memref<f32>
    return %0 : f32
  }
}
