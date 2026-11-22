/* PR #55: multi-block CUDA reduction for SUM/MEAN via atomicAdd.
 * grid.x = output lines, grid.y = chunks; one warp atomic per chunk. */
/* PR #61: single-kernel max/min via atomicCAS on float bits. */
__device__ __forceinline__ float atomicMaxF(float* addr, float v){
  unsigned int* u=(unsigned int*)addr;
  unsigned int old=*u, assumed;
  do { assumed=old;
    if (v <= __int_as_float(assumed)) break;
    old=atomicCAS(u,assumed,__float_as_int(v));
  } while (assumed!=old);
  return __int_as_float(old);
}
__device__ __forceinline__ float atomicMinF(float* addr, float v){
  unsigned int* u=(unsigned int*)addr;
  unsigned int old=*u, assumed;
  do { assumed=old;
    if (v >= __int_as_float(assumed)) break;
    old=atomicCAS(u,assumed,__float_as_int(v));
  } while (assumed!=old);
  return __int_as_float(old);
}
extern "C" __global__ void bmax_atomic(const float* in, float* out,
    int outer, int stride, int inner, int kind, int nblocks) {
  int line=blockIdx.x, chunk=blockIdx.y, tid=threadIdx.x;
  int o=line/inner, t=line%inner;
  int c0=(chunk*stride)/nblocks, c1=((chunk+1)*stride)/nblocks;
  float local=(kind==2)?-3.4028235e38f:3.4028235e38f;
  for(int k=c0+tid;k<c1;k+=blockDim.x){
    float v=in[(o*stride+k)*inner+t];
    local=(kind==2)?fmaxf(local,v):fminf(local,v);
  }
  /* warp reduce, then block reduce in shared: ONE atomic per block (not warp) */
  for(int off=16;off>0;off>>=1){
    float v=__shfl_down_sync(0xffffffff,local,off);
    local=(kind==2)?fmaxf(local,v):fminf(local,v);
  }
  __shared__ float warp[8];
  if((tid&31)==0) warp[tid>>5]=local;
  __syncthreads();
  if(tid<32){
    float v=(tid<8)?warp[tid]:((kind==2)?-3.4028235e38f:3.4028235e38f);
    for(int off=16;off>0;off>>=1){
      float w=__shfl_down_sync(0xffffffff,v,off);
      v=(kind==2)?fmaxf(v,w):fminf(v,w);
    }
    if(tid==0){ if(kind==2) atomicMaxF(&out[line],v); else atomicMinF(&out[line],v); }
  }
}

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
