/* Internal structures for the compact runtime. Not part of the public ABI. */
#ifndef RT_INTERNAL_H
#define RT_INTERNAL_H

#include "bitsy_rt/bitsy_rt.h"

#define RT_MAXDIMS 4
#define RT_MAXIN   8
#define RT_MAXINST 256
#define RT_MAXBUF  256
#define RT_MAX_CAPTURE_SLOTS 64

/* Fused-kernel instruction set (also mirrored in fused.comp). */
typedef enum {
  I_LOAD=0, I_ADD, I_SUB, I_MUL, I_DIV, I_NEG,
  I_EXP, I_LOG, I_SQRT, I_TANH, I_SIGM, I_RELU, I_GELU, I_GT0, I_GELUB,
  /* PR #2 additions */
  I_POW, I_MAXIMUM, I_MINIMUM, I_ABS, I_SIGN,
  I_GT, I_GE, I_LT, I_LE, I_EQ, I_NE, I_WHERE
} RtInstOp;

typedef struct {
  int op;      /* RtInstOp */
  int a;       /* I_LOAD: input slot; unary/binary: stack index */
  int b;       /* binary: stack index */
  float c;     /* I_CONST immediate (unused for now) */
} RtInst;

typedef struct RtBuf {
  int device;
  int n;
  int ndim;
  int64_t dims[RT_MAXDIMS];
  int64_t stride[RT_MAXDIMS];   /* byte-element strides; 0 means broadcast */
  float* host;                  /* CPU storage (owned) */
  void* vk;                     /* Vulkan buffer handle (opaque) */
  void* cap_slot;               /* capture slot holding this buf's replay output */
  int dirty;                    /* output changed on device; download before host read */
  int host_valid;               /* host[] holds real data to upload (leaf), else output scratch */
  int ref;
} RtBuf;

struct RTTensor {
  RT* ctx;
  RTOp op;
  int ndim;
  int64_t dims[RT_MAXDIMS];
  int64_t stride[RT_MAXDIMS];   /* view stride; computed for leaves */
  RTTensor* src[3];
  int nsrc;
  float scalar;
  int axis;
  int keepdim;
  int requires_grad;
  RTTensor* grad;
  RtBuf* buf;                   /* non-NULL once realized */
  int nconsumer;
  uint64_t id;
};

/* A fused pointwise program plus its realized leaf inputs. */
typedef struct {
  RtInst inst[RT_MAXINST];
  int ninst;
  int ninputs;
  RTTensor* in_tensor[RT_MAXIN];
  RtBuf* in_buf[RT_MAXIN];
  int64_t in_stride[RT_MAXIN][RT_MAXDIMS];
  int ndim;
  int64_t dims[RT_MAXDIMS];
  /* PR #7: specialized-program spec constants; sp_ok=1 if translatable */
  int sp_ok;
  int sp[25];
} RtFusion;

/* Vulkan backend state (defined in rt_vk.c). */
typedef struct RtVk RtVk;

struct RT {
  int device;
  char last_error[512];
  int dispatch_count;
  int capture;                  /* capture/replay state machine */
  RtFusion capture_fusion[RT_MAXBUF];
  int capture_n;
  RTTensor* capture_src[RT_MAXBUF];
  int capture_nsrc;
  RtBuf* capture_outputs[RT_MAXBUF];
  int capture_nout;
  RTTensor** nodes;
  int nnodes;
  int capnodes;
  uint64_t next_id;
  RtVk* vk;
};

/* rt.c -> rt_vk.c */
void rt_vk_init(RT*);
void rt_vk_shutdown(RT*);
void rt_vk_fused(RT*, RtFusion*, RtBuf* out);
void rt_vk_gemm(RT*, RtBuf* a, RtBuf* b, RtBuf* bias,
                int trans_a, int trans_b, RtBuf* out);
void rt_vk_reduce(RT*, RtBuf* in, int kind, int axis, RtBuf* out);
void rt_vk_capture_begin(RT*);
void rt_vk_capture_end(RT*);
void rt_vk_replay(RT*);
void rt_vk_submit_only(RT*);
void rt_vk_submit_async(RT*);
void rt_vk_download_to(RT*, RtBuf*, float*);
const float* rt_vk_read_ptr(RT*, RtBuf*, int);
void rt_vk_download_outputs(RT*);
void rt_vk_upload(RT*, RtBuf* b, const float* data);   /* refresh a leaf's device data */
void rt_vk_download(RT*, RtBuf* b);                    /* copy device result back to b->host */
void rt_vk_download_lazy(RT*, RtBuf* b);               /* download once if dirty */

/* rt.c internal helpers */
void rt_errf(RT*, const char* fmt, ...);
RtBuf* rt_buf_new(RT*, int ndim, const int64_t* dims, int device);
void rt_buf_release(RtBuf*);
int64_t rt_numel(int ndim, const int64_t* dims);

#endif
