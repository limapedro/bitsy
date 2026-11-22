/* PR #2 unit tests: GEMM bias/transpose, reduce variants, new pointwise ops. */
#include "bitsy_rt.h"
#include <math.h>
#include <stdio.h>

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("[FAIL] %s\n", msg); fails++; } \
  else printf("[PASS] %s\n", msg); } while (0)
static float md(const float* a, const float* b, int n) {
  float m=0.f; for(int i=0;i<n;++i){float d=fabsf(a[i]-b[i]); if(d>m)m=d;} return m;
}
static void run(RT* c, const char* tag) {
  printf("-- %s --\n", tag);
  /* GEMM with bias: A(2,3) @ B(3,2) + bias(2) */
  int64_t sa[2]={2,3}, sb[2]={3,2}, sbi[1]={2};
  float a[6]={1,2,3,4,5,6}, b[6]={7,8,9,10,11,12}, bi[2]={1,2};
  RTTensor* A=rt_from_host(c,a,2,sa,0),*B=rt_from_host(c,b,2,sb,0),*BI=rt_from_host(c,bi,1,sbi,0);
  RTTensor* Y=rt_gemm(c,A,B,BI,0,0);
  float o[4]={0}; rt_to_host(Y,o);
  CHECK(md(o,(float[]){59,66,140,156},4)<1e-4f,"gemm+bias");

  /* transpose flags: A(3,2) stored row-major, A^T(2,3) @ B(3,2) */
  int64_t sat[2]={3,2};
  float at[6]={1,4,2,5,3,6}; /* A^T = [[1,2,3],[4,5,6]] stored as (3,2): rows [1,4],[2,5],[3,6] */
  RTTensor* AT=rt_from_host(c,at,2,sat,0);
  RTTensor* Yt=rt_gemm(c,AT,B,NULL,1,0);
  float ot[4]={0}; rt_to_host(Yt,ot);
  CHECK(md(ot,(float[]){58,64,139,154},4)<1e-4f,"gemm transA");

  /* reduce mean / max / min */
  int64_t sv[1]={6}; float v[6]={1,2,3,4,5,6};
  RTTensor* V=rt_from_host(c,v,1,sv,0);
  float rm[1]; rt_to_host(rt_reduce(c,V,RT_REDUCE_MEAN,0,0),rm);
  CHECK(md(rm,(float[]){3.5f},1)<1e-5f,"reduce mean");
  float rx[1]; rt_to_host(rt_reduce(c,V,RT_REDUCE_MAX,0,0),rx);
  CHECK(md(rx,(float[]){6.f},1)<1e-5f,"reduce max");
  float rn[1]; rt_to_host(rt_reduce(c,V,RT_REDUCE_MIN,0,0),rn);
  CHECK(md(rn,(float[]){1.f},1)<1e-5f,"reduce min");

  /* pointwise: pow, max, min, abs, comparisons, where */
  int64_t s2[1]={4}; float p[4]={2,3,4,5}, q[4]={3,2,5,1};
  RTTensor* P=rt_from_host(c,p,1,s2,0),*Q=rt_from_host(c,q,1,s2,0);
  float op[4]; rt_to_host(rt_pow(c,P,Q),op);
  CHECK(md(op,(float[]){8,9,1024,5},4)<1e-3f,"pow");
  float om[4]; rt_to_host(rt_maximum(c,P,Q),om);
  CHECK(md(om,(float[]){3,3,5,5},4)<1e-5f,"maximum");
  float oi[4]; rt_to_host(rt_minimum(c,P,Q),oi);
  CHECK(md(oi,(float[]){2,2,4,1},4)<1e-5f,"minimum");
  float neg[4]={-1,2,-3,4}; RTTensor* N=rt_from_host(c,neg,1,s2,0);
  float oa[4]; rt_to_host(rt_abs(c,N),oa);
  CHECK(md(oa,(float[]){1,2,3,4},4)<1e-5f,"abs");
  float ogt[4]; rt_to_host(rt_gt(c,P,Q),ogt);
  CHECK(md(ogt,(float[]){0,1,0,1},4)<1e-5f,"gt");
  RTTensor* cond=rt_gt(c,P,Q);
  float ow[4]; rt_to_host(rt_where(c,cond,P,Q),ow);
  CHECK(md(ow,(float[]){3,3,5,5},4)<1e-5f,"where");

  /* ---- autograd: gradients of all primitives ---- */
  int64_t s3[1]={3}; float gx[3]={2,3,4};
  RTTensor* X=rt_from_host(c,gx,1,s3,1);
  RTTensor* sum=rt_reduce(c,X,RT_REDUCE_SUM,0,0);
  rt_backward(c,sum);
  float gg[3]={0}; rt_to_host(rt_grad(X),gg);
  CHECK(md(gg,(float[]){1,1,1},3)<1e-5f,"grad sum");

  RTTensor* X2=rt_from_host(c,gx,1,s3,1);
  RTTensor* mean=rt_reduce(c,X2,RT_REDUCE_MEAN,0,0);
  rt_backward(c,mean);
  float gm[3]={0}; rt_to_host(rt_grad(X2),gm);
  CHECK(md(gm,(float[]){1.f/3,1.f/3,1.f/3},3)<1e-5f,"grad mean");

  RTTensor* X3=rt_from_host(c,gx,1,s3,1);
  RTTensor* mx=rt_reduce(c,X3,RT_REDUCE_MAX,0,0);
  rt_backward(c,mx);
  float gmx[3]={0}; rt_to_host(rt_grad(X3),gmx);
  CHECK(md(gmx,(float[]){0,0,1},3)<1e-5f,"grad max");

  RTTensor* X4=rt_from_host(c,gx,1,s3,1);
  RTTensor* sq=rt_mul(c,X4,X4);
  RTTensor* loss=rt_reduce(c,sq,RT_REDUCE_SUM,0,0);
  rt_backward(c,loss);
  float gsq[3]={0}; rt_to_host(rt_grad(X4),gsq);
  CHECK(md(gsq,(float[]){4,6,8},3)<1e-4f,"grad chain x^2");
}

int main(void) {
  RT* c=rt_create(RT_CPU); run(c,"CPU");
#ifdef RT_WITH_VULKAN
  RT* cv=rt_create(RT_VULKAN); run(cv,"Vulkan"); rt_destroy(cv);
#endif
  rt_destroy(c);
  printf("%s: %d failures\n", fails?"FAILED":"OK", fails);
  return fails?1:0;
}
