/* Compact runtime unit tests (PR #1): fusion count + numeric sanity + grads. */
#include "bitsy_rt.h"
#include <math.h>
#include <stdio.h>

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("[FAIL] %s\n", msg); fails++; } \
  else printf("[PASS] %s\n", msg); } while (0)

static float maxdiff(const float* a, const float* b, int n) {
  float m = 0.f;
  for (int i=0;i<n;++i){ float d=fabsf(a[i]-b[i]); if(d>m)m=d; }
  return m;
}

int main(void) {
  RT* c = rt_create(RT_CPU);
  CHECK(c != NULL, "context create");

  /* ---- fusion: ((t*t)+t).relu() must run in ONE dispatch ---- */
  int64_t s[2] = {2,3};
  float x[6] = {1,2,3,4,5,6};
  RTTensor* t = rt_from_host(c, x, 2, s, 0);
  CHECK(rt_ndim(t)==2, "tensor ndim");
  RTTensor* m2 = rt_mul(c, t, t);
  RTTensor* y  = rt_add(c, m2, t);
  RTTensor* z  = rt_relu(c, y);
  rt_reset_dispatch_count(c);
  float out[6] = {0};
  rt_to_host(z, out);
  CHECK(rt_dispatch_count(c) == 1, "pointwise chain fuses into 1 dispatch");
  float want[6] = {2,6,12,20,30,42};
  CHECK(maxdiff(out, want, 6) < 1e-5f, "pointwise numerics");

  /* ---- matmul (M,K)x(K,N) ---- */
  int64_t sa[2]={2,3}, sb[2]={3,2};
  float a[6]={1,2,3,4,5,6}, b[6]={7,8,9,10,11,12};
  RTTensor* A=rt_from_host(c,a,2,sa,0), *B=rt_from_host(c,b,2,sb,0);
  RTTensor* M=rt_matmul(c,A,B);
  float mo[4]={0}; rt_to_host(M,mo);
  float mw[4]={58,64,139,154};
  CHECK(maxdiff(mo,mw,4)<1e-4f, "matmul numerics");

  /* ---- reduce sum ---- */
  RTTensor* R=rt_reduce_sum(c, t, 1, 0);
  float ro[2]={0}; rt_to_host(R,ro);
  CHECK(maxdiff(ro,(float[]){6,15},2)<1e-5f, "reduce_sum numerics");

  /* ---- autograd: d(sum(p*p))/dp = 2p ---- */
  int64_t s1[1]={6};
  RTTensor* p=rt_from_host(c,x,1,s1,1);
  RTTensor* sq=rt_mul(c,p,p);
  RTTensor* loss=rt_reduce_sum(c,sq,0,0);
  rt_backward(c,loss);
  float g[6]={0}; rt_to_host(rt_grad(p),g);
  float gw[6]={2,4,6,8,10,12};
  CHECK(maxdiff(g,gw,6)<1e-4f, "autograd d(sum x^2)=2x");

  printf("version=%s\n", rt_version());
  rt_destroy(c);

#ifdef RT_WITH_VULKAN
  /* ---- Vulkan backend: same fused chain on the GPU ---- */
  printf("-- Vulkan backend --\n");
  RT* cv = rt_create(RT_VULKAN);
  CHECK(cv != NULL, "vulkan context create");
  RTTensor* tv = rt_from_host(cv, x, 2, s, 0);
  RTTensor* mv = rt_mul(cv, tv, tv);
  RTTensor* yv = rt_add(cv, mv, tv);
  RTTensor* zv = rt_relu(cv, yv);
  rt_reset_dispatch_count(cv);
  float ov[6] = {0};
  rt_to_host(zv, ov);
  CHECK(rt_dispatch_count(cv) == 1, "vulkan pointwise chain 1 dispatch");
  CHECK(maxdiff(ov, want, 6) < 1e-5f, "vulkan pointwise numerics");
  RTTensor* Mv = rt_matmul(cv,
    rt_from_host(cv,a,2,sa,0), rt_from_host(cv,b,2,sb,0));
  float mov[4]={0}; rt_to_host(Mv,mov);
  CHECK(maxdiff(mov,mw,4)<1e-4f, "vulkan matmul numerics");
  rt_destroy(cv);
#endif

  printf("%s: %d failures\n", fails?"FAILED":"OK", fails);
  return fails ? 1 : 0;
}
