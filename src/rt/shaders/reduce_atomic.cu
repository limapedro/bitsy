/* PR #55: multi-block CUDA reduction for SUM/MEAN via atomicAdd.
 * grid.x = output lines, grid.y = chunks; one warp atomic per chunk. */
/* PR #60: two-stage max/min reduction (no float atomicMax). */
extern "C" __global__ void bmax1(const float* in, float* part,
    int outer, int stride, int inner, int kind, int npart) {
  int line = blockIdx.x, chunk = blockIdx.y, tid = threadIdx.x;
  int o = line / inner, t = line % inner;
  int c0 = (chunk * stride) / npart, c1 = ((chunk + 1) * stride) / npart;
  float s = (kind == 2) ? -3.4028235e38f : 3.4028235e38f;
  for (int k = c0 + tid; k < c1; k += blockDim.x) {
    float v = in[(o * stride + k) * inner + t];
    s = (kind == 2) ? fmaxf(s, v) : fminf(s, v);
  }
  __shared__ float sh[256];
  sh[tid] = s; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] = (kind == 2) ? fmaxf(sh[tid], sh[tid+off]) : fminf(sh[tid], sh[tid+off]);
    __syncthreads();
  }
  if (tid == 0) part[line * npart + chunk] = sh[0];
}
extern "C" __global__ void bmax2(const float* part, float* out,
    int lines, int npart, int kind) {
  int line = blockIdx.x * blockDim.x + threadIdx.x;
  if (line >= lines) return;
  float s = (kind == 2) ? -3.4028235e38f : 3.4028235e38f;
  for (int i = 0; i < npart; ++i) {
    float v = part[line * npart + i];
    s = (kind == 2) ? fmaxf(s, v) : fminf(s, v);
  }
  out[line] = s;
}

extern "C" __global__ void breduce_sum(const float* in, float* out,
    int outer, int stride, int inner, int nblocks, float scale) {
  int line = blockIdx.x, chunk = blockIdx.y, tid = threadIdx.x;
  int o = line / inner, t = line % inner;
  int c0 = (chunk * stride) / nblocks;
  int c1 = ((chunk + 1) * stride) / nblocks;
  float s = 0.f;
  for (int k = c0 + tid; k < c1; k += blockDim.x)
    s += in[(o * stride + k) * inner + t];
  for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
  if ((tid & 31) == 0) atomicAdd(&out[line], s * scale);
}
