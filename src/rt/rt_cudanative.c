/* PR #50: dynamically loaded CUDA-native fused pointwise execution. */
#include "rt_cudanative.h"
#include "fused_cuda.ptx.h"
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
  CUdptr params; void* host; size_t host_cap;
  void* hostn[4]; int hidx;
  CUdptr out_ring[8]; size_t out_cap[8]; int oidx;
  int dbg;
  CUres (*cuInit)(unsigned); CUres (*cuCtxGetCurrent)(CUctx*);
  CUres (*cuCtxCreate)(CUctx*,unsigned,int); CUres (*cuDeviceGet)(int*,int);
  CUres (*cuModuleLoadData)(CUmod*,const void*);
  CUres (*cuModuleGetFunction)(CUfunc*,CUmod,const char*);
  CUres (*cuMemAlloc)(CUdptr*,size_t); CUres (*cuMemAllocHost)(void**,size_t);
  CUres (*cuMemcpyHtoDAsync)(CUdptr,const void*,size_t,CUstream);
  CUres (*cuMemcpyDtoHAsync)(void*,CUdptr,size_t,CUstream);
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
  L(cuModuleLoadData,"cuModuleLoadData"); L(cuModuleGetFunction,"cuModuleGetFunction");
  L(cuMemAlloc,"cuMemAlloc_v2"); L(cuMemAllocHost,"cuMemAllocHost_v2");
  L(cuMemcpyHtoDAsync,"cuMemcpyHtoDAsync_v2"); L(cuMemcpyDtoHAsync,"cuMemcpyDtoHAsync_v2");
  L(cuLaunchKernel,"cuLaunchKernel"); L(cuStreamCreate,"cuStreamCreate");
  L(cuStreamSynchronize,"cuStreamSynchronize");
  if(s->cuInit(0)!=0){ DBG("init failed"); goto fail; }
  if(s->cuCtxGetCurrent(&s->ctx)!=0||!s->ctx){
    int d=0; if(s->cuDeviceGet(&d,0)!=0||s->cuCtxCreate(&s->ctx,0,d)!=0){ DBG("ctx failed"); goto fail; }
  }
  if(s->cuStreamCreate(&s->stream,1)!=0){ DBG("stream failed"); goto fail; }
  if(s->cuModuleLoadData(&s->mod,rt_fused_cuda_ptx)!=0){ DBG("module failed"); goto fail; }
  if(s->cuModuleGetFunction(&s->fn,s->mod,"bfused")!=0){ DBG("function failed"); goto fail; }
  s->params=0; s->host_cap=64u<<20;
  if(s->cuMemAllocHost(&s->host,s->host_cap)!=0){ DBG("host failed"); goto fail; }
  for(int i=0;i<4;++i) s->hostn[i]=(char*)s->host+(size_t)(s->host_cap/4)*i;
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

void* rt_cn_out(size_t bytes){
  struct CN* s=cn_init(); if(!s) return NULL;
  int i=s->oidx&7; s->oidx++;
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
  void* h=s->hostn[s->hidx&3]; s->hidx++;
  if(s->dbg) fprintf(stderr,"[cn] ptr dptr=%p host=%p bytes=%zu\n",dptr,h,bytes);
  if(s->cuMemcpyDtoHAsync(h,(CUdptr)(size_t)dptr,bytes,s->stream)!=0) return NULL;
  if(s->cuStreamSynchronize(s->stream)!=0) return NULL;
  return (const float*)h;
}
void rt_cn_flush(void){ if(G&&G->stream) G->cuStreamSynchronize(G->stream); }
