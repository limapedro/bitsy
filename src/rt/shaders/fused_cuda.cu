/* PR #50: CUDA-native fused pointwise kernel (mirror of fused.comp).
 * Same params layout: [0]=n [1]=ninst [2]=ninputs [3]=ndim
 * [4..7]=out dims, [8+a*4+j]=input strides, [40]=out addr (u64),
 * [42+slot*2]=input addr (u64), [58+k*4]=op,a,b,c-bits.
 */
extern "C" __global__ void bfused(const int* p, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p[0]) return;
  int nd = p[3];
  int coord[4];
  int rem = i;
  for (int j = nd - 1; j >= 0; --j) { coord[j] = rem % p[4 + j]; rem /= p[4 + j]; }
  float stk[16];
  int sp = 0;
  for (int k = 0; k < p[1]; ++k) {
    int base = 58 + k * 4;
    int op = p[base], a = p[base + 1], b = p[base + 2];
    float c = __int_as_float(p[base + 3]);
    float va = sp > a ? stk[a] : 0.f;
    float vb = sp > b ? stk[b] : 0.f;
    if (op == 0) {
      const float* in = (const float*)(unsigned long long)(
        (unsigned int)p[42 + a * 2] | ((unsigned long long)(unsigned int)p[43 + a * 2] << 32));
      long long off = 0;
      for (int j = 0; j < nd; ++j) off += (long long)coord[j] * (long long)p[8 + a * 4 + j];
      stk[sp++] = in[off];
    } else if (op == 1) stk[sp++] = va + vb;
    else if (op == 2) stk[sp++] = va - vb;
    else if (op == 3) stk[sp++] = va * vb;
    else if (op == 4) stk[sp++] = vb != 0.f ? va / vb : 0.f;
    else if (op == 5) stk[sp++] = -va;
    else if (op == 6) stk[sp++] = __expf(va);
    else if (op == 7) stk[sp++] = logf(va);
    else if (op == 8) stk[sp++] = sqrtf(fmaxf(va, 0.f));
    else if (op == 9) stk[sp++] = tanhf(va);
    else if (op == 10) stk[sp++] = 1.f / (1.f + __expf(-va));
    else if (op == 11) stk[sp++] = fmaxf(va, 0.f);
    else if (op == 12 || op == 14) {
      float s = va < 0.f ? -1.f : 1.f, ax = fabsf(va) * 0.70710678f;
      float t = 1.f / (1.f + 0.3275911f * ax);
      float pp = ((((1.061405429f*t - 1.453152027f)*t + 1.421413741f)*t
                   - 0.284496736f)*t + 0.254829592f)*t;
      float e = s * (1.f - pp * __expf(-ax * ax));
      stk[sp++] = va * 0.5f * (1.f + e);
    }
    else if (op == 13) stk[sp++] = va > 0.f ? 1.f : 0.f;
    else if (op == 15) stk[sp++] = powf(va, vb);
    else if (op == 16) stk[sp++] = fmaxf(va, vb);
    else if (op == 17) stk[sp++] = fminf(va, vb);
    else if (op == 18) stk[sp++] = fabsf(va);
    else if (op == 19) stk[sp++] = va > 0.f ? 1.f : (va < 0.f ? -1.f : 0.f);
    else if (op == 20) stk[sp++] = va > vb ? 1.f : 0.f;
    else if (op == 21) stk[sp++] = va >= vb ? 1.f : 0.f;
    else if (op == 22) stk[sp++] = va < vb ? 1.f : 0.f;
    else if (op == 23) stk[sp++] = va <= vb ? 1.f : 0.f;
    else if (op == 24) stk[sp++] = va == vb ? 1.f : 0.f;
    else if (op == 25) stk[sp++] = va != vb ? 1.f : 0.f;
    else if (op == 26) stk[sp++] = c;   /* WHERE constant fallback */
    else stk[sp++] = va;
    (void)c;
  }
  out[i] = stk[sp - 1];
}
