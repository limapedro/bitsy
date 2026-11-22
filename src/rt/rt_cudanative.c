/* PR #50: dynamically loaded CUDA-native fused pointwise execution. */
#include "rt_cudanative.h"
#include "fused_cuda.ptx.h"
#include "reduce_atomic.ptx.h"
#include "gemm_cuda.ptx.h"
#include "norm_cuda.ptx.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#define LLOAD() LoadLibraryA("nvcuda.dll")
#define LSYM(h,n) GetProcAddress((HMODULE)(h),n)
#define LFREE(h) FreeLibrary((HMODULE)(h))
#else
#include <dlfcn.h>
#define LLOAD() dlopen("libcuda.so.1",RTLD_LAZY)
#define LSYM(h,n) dlsym((void*)(h),n)
#define LFREE(h) dlclose((void*)(h))
#endif

typedef int CUres; typedef void* CUctx; typedef void* CUstream;
typedef void* CUmod; typedef void* CUfunc; typedef unsigned long long CUdptr;
struct CN {
  void* lib; CUctx ctx; CUstream stream; CUmod mod; CUfunc fn;
  CUmod mod3; CUfunc f_sum; CUfunc f_max1, f_max2;
  CUmod mod4; CUfunc f_gemm_tiled;
  CUmod mod5; CUfunc f_softmax, f_layernorm;
  CUdptr params; void* host; size_t host_cap;
  void* hostn[256]; int hidx;
  CUdptr out_ring[256]; size_t out_cap[256]; int oidx;
  CUdptr scratch; size_t scratch_cap;
  int dbg;
  CUres (*cuInit)(unsigned); CUres (*cuCtxGetCurrent)(CUctx*);
  CUres (*cuCtxCreate)(CUctx*,unsigned,int); CUres (*cuDeviceGet)(int*,int);
  CUres (*cuPrimaryRetain)(CUctx*,int);
  CUres (*cuModuleLoadData)(CUmod*,const void*);
  CUres (*cuModuleGetFunction)(CUfunc*,CUmod,const char*);
  CUres (*cuMemAlloc)(CUdptr*,size_t); CUres (*cuMemAllocHost)(void**,size_t);
  CUres (*cuMemcpyHtoDAsync)(CUdptr,const void*,size_t,CUstream);
  CUres (*cuMemcpyDtoHAsync)(void*,CUdptr,size_t,CUstream);
  CUres (*cuMemsetD8Async)(CUdptr,unsigned char,size_t,CUstream);
  CUres (*cuLaunchKernel)(CUfunc,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,
                          unsigned,CUstream,void**,void**);
  CUres (*cuStreamCreate)(CUstream*,unsigned); CUres (*cuStreamSynchronize)(CUstream);
};
static struct CN* G=NULL;
#define L(f,n) do{ s->f=(void*)LSYM(lib,n); if(!s->f) goto fail; }while(0)
#define DBG(x) do{ if(s->dbg) fprintf(stderr,"[cn] " x "\n"); }while(0)

static struct CN* cn_init(void){
  if(G) return G;
  void* lib=LLOAD(); if(!lib) return NULL;
  struct CN* s=(struct CN*)calloc(1,sizeof(*s)); if(!s){ LFREE(lib); return NULL; }
  s->lib=lib; s->dbg=getenv("BT_CN_DBG")!=NULL;
  L(cuInit,"cuInit"); L(cuCtxGetCurrent,"cuCtxGetCurrent");
  L(cuCtxCreate,"cuCtxCreate_v2"); L(cuDeviceGet,"cuDeviceGet");
  L(cuPrimaryRetain,"cuDevicePrimaryCtxRetain");
  L(cuModuleLoadData,"cuModuleLoadData"); L(cuModuleGetFunction,"cuModuleGetFunction");
  L(cuMemAlloc,"cuMemAlloc_v2"); L(cuMemAllocHost,"cuMemAllocHost_v2");
  L(cuMemcpyHtoDAsync,"cuMemcpyHtoDAsync_v2"); L(cuMemcpyDtoHAsync,"cuMemcpyDtoHAsync_v2");
  L(cuMemsetD8Async,"cuMemsetD8Async");
  L(cuLaunchKernel,"cuLaunchKernel"); L(cuStreamCreate,"cuStreamCreate");
  L(cuStreamSynchronize,"cuStreamSynchronize");
  if(s->cuInit(0)!=0){ DBG("init failed"); goto fail; }
  if(s->cuCtxGetCurrent(&s->ctx)!=0||!s->ctx){
    int d=0; if(s->cuDeviceGet(&d,0)!=0){ DBG("device failed"); goto fail; }
    /* Use the PRIMARY context (shared with PyTorch/CUDA runtime) so creating
     * a context here does not cost ~0.4ms/call later. */
    if(s->cuPrimaryRetain(&s->ctx,d)!=0){ DBG("primary ctx failed"); goto fail; }
  }
  if(s->cuStreamCreate(&s->stream,1)!=0){ DBG("stream failed"); goto fail; }
  if(s->cuModuleLoadData(&s->mod,rt_fused_cuda_ptx)!=0){ DBG("module failed"); goto fail; }
  if(s->cuModuleGetFunction(&s->fn,s->mod,"bfused")!=0){ DBG("function failed"); goto fail; }
  if(s->cuModuleLoadData(&s->mod3,rt_reduce_atomic_ptx)==0) {
    s->cuModuleGetFunction(&s->f_sum,s->mod3,"breduce_sum");
    s->cuModuleGetFunction(&s->f_max1,s->mod3,"bmax1");
    s->cuModuleGetFunction(&s->f_max2,s->mod3,"bmax2");
  }
  if(s->cuModuleLoadData(&s->mod4,rt_gemm_cuda_ptx)==0)
    s->cuModuleGetFunction(&s->f_gemm_tiled,s->mod4,"bgemm_tiled");
  if(s->cuModuleLoadData(&s->mod5,rt_norm_cuda_ptx)==0) {
    s->cuModuleGetFunction(&s->f_softmax,s->mod5,"bsoftmax");
    s->cuModuleGetFunction(&s->f_layernorm,s->mod5,"blayernorm");
  }
  s->params=0; s->host_cap=64u<<20;
  if(s->cuMemAllocHost(&s->host,s->host_cap)!=0){ DBG("host failed"); goto fail; }
  for(int i=0;i<256;++i) s->hostn[i]=(char*)s->host+(size_t)(s->host_cap/256)*i;
  DBG("ready"); G=s; return s;
fail:
  LFREE(lib); free(s); return NULL;
}
int rt_cn_available(void){ return cn_init()!=NULL; }
void* rt_cn_alloc(size_t bytes){
  struct CN* s=cn_init(); if(!s) return NULL;
  CUdptr d=0; if(s->cuMemAlloc(&d,bytes)!=0) return NULL;
  return (void*)(size_t)d;
}
int rt_cn_upload(void* dptr,const float* host,size_t bytes){
  struct CN* s=cn_init(); if(!s||!dptr) return -1;
  return s->cuMemcpyHtoDAsync((CUdptr)(size_t)dptr,host,bytes,s->stream);
}

void* rt_cn_scratch(size_t bytes){
  struct CN* s=cn_init(); if(!s) return NULL;
  if(bytes>s->scratch_cap){
    if(s->cuMemAlloc(&s->scratch,bytes)!=0) return NULL;
    s->scratch_cap=bytes;
  }
  return (void*)(size_t)s->scratch;
}
void* rt_cn_out(size_t bytes){
  struct CN* s=cn_init(); if(!s) return NULL;
  int i=s->oidx&255; s->oidx++;
  if(bytes>s->out_cap[i]){
    if(s->cuMemAlloc(&s->out_ring[i],bytes)!=0) return NULL;
    s->out_cap[i]=bytes;
  }
  return (void*)(size_t)s->out_ring[i];
}
void* rt_cn_run(const int* p, int pwords, int n, void* out){
  struct CN* s=cn_init(); if(!s||!out) return NULL;
  if(!s->params){ if(s->cuMemAlloc(&s->params,64*1024)!=0) return NULL; }
  if(s->dbg) fprintf(stderr,"[cn] run params=%p out=%p p=%p words=%d\n",(void*)(size_t)s->params,out,p,pwords);
  if(s->cuMemcpyHtoDAsync(s->params,p,(size_t)pwords*4,s->stream)!=0) return NULL;
  CUdptr pp=s->params, oo=(CUdptr)(size_t)out;
  void* args[2]={ (void*)&pp, (void*)&oo };
  int blocks=(n+255)/256;
  if(s->cuLaunchKernel(s->fn,blocks,1,1,256,1,1,0,s->stream,args,NULL)!=0) return NULL;
  return out;
}
int rt_cn_read(const void* dptr, float* host, size_t bytes){
  struct CN* s=G; if(!s||!dptr||bytes>s->host_cap) return -1;
  if(s->cuMemcpyDtoHAsync(s->host,(CUdptr)(size_t)dptr,bytes,s->stream)!=0) return -2;
  if(s->cuStreamSynchronize(s->stream)!=0) return -3;
  memcpy(host,s->host,bytes);
  return 0;
}
const float* rt_cn_ptr(const void* dptr,size_t bytes){
  struct CN* s=G; if(!s||!dptr||bytes>s->host_cap/4) return NULL;
  void* h=s->hostn[s->hidx&255]; s->hidx++;
  if(s->dbg) fprintf(stderr,"[cn] ptr dptr=%p host=%p bytes=%zu\n",dptr,h,bytes);
  if(s->cuMemcpyDtoHAsync(h,(CUdptr)(size_t)dptr,bytes,s->stream)!=0) return NULL;
  if(s->cuStreamSynchronize(s->stream)!=0) return NULL;
  return (const float*)h;
}
int rt_cn_reduce_fast(void* in,void* out,int outer,int stride,int inner,int kind){
  struct CN* s=cn_init(); if(!s||!s->f_sum||!in||!out) return -1;
  int lines=outer*inner;
  size_t bytes=sizeof(float)*(size_t)lines;
  /* Zero on-device with a stream-ordered memset: never touch the pinned host
   * buffer that may back a live zero-copy numpy view (gradcheck corruption). */
  if(s->cuMemsetD8Async((CUdptr)(size_t)out,0,bytes,s->stream)!=0) return -2;
  int nblocks = stride > 8192 ? 1024 : (stride > 512 ? 64 : 1);
  if(nblocks>stride) nblocks=stride>0?stride:1;
  float scale=(kind==1)?(1.f/(float)stride):1.f;
  CUdptr i=(CUdptr)(size_t)in,o=(CUdptr)(size_t)out;
  void* args[6]={(void*)&i,(void*)&o,(void*)&outer,(void*)&stride,(void*)&inner,(void*)&nblocks};
  /* scale is appended via a second array below; kernel has 7 params */
  float sc2=scale; (void)sc2;
  void* args7[7]={(void*)&i,(void*)&o,(void*)&outer,(void*)&stride,(void*)&inner,(void*)&nblocks,(void*)&scale};
  if(s->cuLaunchKernel(s->f_sum,(unsigned)lines,(unsigned)nblocks,1,256,1,1,
      0,s->stream,args7,NULL)!=0) return -4;
  (void)args;
  return 0;
}
int rt_cn_reduce_max(void* in,void* part,void* out,int outer,int stride,int inner,int kind){
  struct CN* s=cn_init(); if(!s||!s->f_max1||!s->f_max2||!in||!part||!out) return -1;
  int lines=outer*inner, np=64;
  CUdptr i=(CUdptr)(size_t)in,p=(CUdptr)(size_t)part,o=(CUdptr)(size_t)out;
  void* a1[7]={(void*)&i,(void*)&p,(void*)&outer,(void*)&stride,(void*)&inner,(void*)&kind,(void*)&np};
  if(s->cuLaunchKernel(s->f_max1,(unsigned)lines,(unsigned)np,1,256,1,1,0,s->stream,a1,NULL)!=0) return -2;
  void* a2[4]={(void*)&p,(void*)&o,(void*)&lines,(void*)&np};
  if(s->cuLaunchKernel(s->f_max2,(unsigned)((lines+255)/256),1,1,256,1,1,0,s->stream,a2,NULL)!=0) return -3;
  return 0;
}
int rt_cn_gemm_fast(void* A,void* B,void* bias,void* C,int M,int N,int K,int flags){
  struct CN* s=cn_init(); if(!s||!s->f_gemm_tiled||!A||!B||!C) return -1;
  CUdptr a=(CUdptr)(size_t)A,b=(CUdptr)(size_t)B,
         bi=(CUdptr)(size_t)(bias?bias:A),c=(CUdptr)(size_t)C;
  void* args[8]={(void*)&a,(void*)&b,(void*)&bi,(void*)&c,
                 (void*)&M,(void*)&N,(void*)&K,(void*)&flags};
  if(s->cuLaunchKernel(s->f_gemm_tiled,(unsigned)((N+15)/16),(unsigned)((M+15)/16),1,
      16,16,1,0,s->stream,args,NULL)!=0) return -2;
  return 0;
}
int rt_cn_softmax(void* in,void* bias,void* out,int rows,int cols,int kind,int has_bias){
  struct CN* s=cn_init(); if(!s||!s->f_softmax||!in||!out) return -1;
  CUdptr i=(CUdptr)(size_t)in,bi=(CUdptr)(size_t)(bias?bias:in),o=(CUdptr)(size_t)out;
  void* args[7]={(void*)&i,(void*)&bi,(void*)&o,(void*)&rows,(void*)&cols,(void*)&kind,(void*)&has_bias};
  if(s->cuLaunchKernel(s->f_softmax,(unsigned)rows,1,1,256,1,1,
      (unsigned)(cols*sizeof(float)),s->stream,args,NULL)!=0) return -2;
  return 0;
}
int rt_cn_layernorm(void* in,void* w,void* b,void* res,void* out,
                    int rows,int cols,float eps,int has_res){
  struct CN* s=cn_init(); if(!s||!s->f_layernorm||!in||!w||!b||!out) return -1;
  CUdptr i=(CUdptr)(size_t)in,wi=(CUdptr)(size_t)w,bi=(CUdptr)(size_t)b,
         ri=(CUdptr)(size_t)(res?res:in),o=(CUdptr)(size_t)out;
  void* args[9]={(void*)&i,(void*)&wi,(void*)&bi,(void*)&ri,(void*)&o,
                 (void*)&rows,(void*)&cols,(void*)&eps,(void*)&has_res};
  if(s->cuLaunchKernel(s->f_layernorm,(unsigned)rows,1,1,256,1,1,
      (unsigned)(cols*sizeof(float)),s->stream,args,NULL)!=0) return -2;
  return 0;
}
void rt_cn_flush(void){ if(G&&G->stream) G->cuStreamSynchronize(G->stream); }
