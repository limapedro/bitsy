/* PR #58: tiled CUDA GEMM with shared memory (16x16 tiles, 256 threads).
 * flags: bit0 transA, bit1 transB, bit2 bias, bit3 relu. */
#define T 16
extern "C" __global__ void bgemm_tiled(const float* A, const float* B,
    const float* bias, float* C, int M, int N, int K, int flags) {
  __shared__ float As[T][T+1], Bs[T][T+1];
  int row = blockIdx.y * T + threadIdx.y;
  int col = blockIdx.x * T + threadIdx.x;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    int ak = k0 + threadIdx.x;
    As[threadIdx.y][threadIdx.x] =
      (row < M && ak < K) ? ((flags & 1) ? A[ak * M + row] : A[row * K + ak]) : 0.f;
    int bk = k0 + threadIdx.y;
    Bs[threadIdx.y][threadIdx.x] =
      (bk < K && col < N) ? ((flags & 2) ? B[col * K + bk] : B[bk * N + col]) : 0.f;
    __syncthreads();
    for (int k = 0; k < T; ++k) acc = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
    __syncthreads();
  }
  if (row < M && col < N) {
    if (flags & 4) acc += bias[col];
    if (flags & 8) acc = fmaxf(acc, 0.f);
    C[row * N + col] = acc;
  }
}
