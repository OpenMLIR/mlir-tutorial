mkdir build; cd build
# 请根据实际情况修改
export export LLVM_BUILD_DIR=~/llvm-project/build
cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Debug
ninja
