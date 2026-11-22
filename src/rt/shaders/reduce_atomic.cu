/* PR #55: multi-block CUDA reduction for SUM/MEAN via atomicAdd.
 * grid.x = output lines, grid.y = chunks; one warp atomic per chunk. */
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
