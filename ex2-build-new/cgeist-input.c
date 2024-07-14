// cgeist run cmd
// cgeist ArraySum.c --function=* -S --memref-fullrank -o test.mlir

#define N 10

float ArraySum(float a[N]) {
// Polygeist的pragma，会优先生成affine
#pragma scop
  float sum = 0;
  for (int i = 0; i < N; i++) {
    sum += a[i];
  }
  return sum;
// Polygeist的pragma，会优先生成affine
#pragma endscop
}
