/* PR #59: CUDA softmax/log_softmax and LayerNorm (one block per row),
 * mirroring softmax.comp / layernorm.comp so mixed graphs stay on one backend. */
extern "C" __global__ void bsoftmax(const float* in, const float* bias,
    float* out, int rows, int cols, int kind, int has_bias) {
  extern __shared__ float smem[];
  int row = blockIdx.x, tid = threadIdx.x;
  if (row >= rows) return;
  float lmax = -3.4028235e38f;
  for (int i = tid; i < cols; i += blockDim.x) {
    float v = in[row * cols + i] + (has_bias ? bias[i] : 0.f);
    smem[i] = v; lmax = fmaxf(lmax, v);
  }
  __shared__ float red[256];
  red[tid] = lmax; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  float mx = red[0];
  float lsum = 0.f;
  for (int i = tid; i < cols; i += blockDim.x) {
    float e = __expf(smem[i] - mx); smem[i] = e; lsum += e;
  }
  red[tid] = lsum; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float s = red[0] + 1e-20f;
  for (int i = tid; i < cols; i += blockDim.x) {
    if (kind == 0) out[row * cols + i] = smem[i] / s;
    else out[row * cols + i] = (smem[i] > 0.f) ? logf(smem[i] / s) : -3.4028235e38f;
  }
}

extern "C" __global__ void blayernorm(const float* in, const float* W,
    const float* B, const float* res, float* out,
    int rows, int cols, float eps, int has_res) {
  extern __shared__ float smem[];
  __shared__ float red[256];
  int row = blockIdx.x, tid = threadIdx.x;
  if (row >= rows) return;
  for (int i = tid; i < cols; i += blockDim.x)
    smem[i] = in[row * cols + i] + (has_res ? res[row * cols + i] : 0.f);
  __syncthreads();
  float s = 0.f;
  for (int i = tid; i < cols; i += blockDim.x) s += smem[i];
  red[tid] = s; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float mean = red[0] / (float)cols;
  float v = 0.f;
  for (int i = tid; i < cols; i += blockDim.x) { float d = smem[i] - mean; smem[i] = d; v += d * d; }
  red[tid] = v; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float scale = rsqrtf(red[0] / (float)cols + eps);
  for (int i = tid; i < cols; i += blockDim.x)
    out[row * cols + i] = smem[i] * scale * W[i] + B[i];
}
