/* Bitsy 0.2 compact runtime — graph, scheduler/fusion, CPU exec, autograd.
 * tinygrad layering in C: frontend builds nodes; realize schedules fused
 * programs; executor runs them; backward walks the graph. No per-op dispatch.
 */
#include "rt_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RT_VERSION "0.2.0-compact"

void rt_errf(RT* c, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  vsnprintf(c->last_error, sizeof(c->last_error), fmt, ap);
  va_end(ap);
}

int64_t rt_numel(int nd, const int64_t* d) {
  int64_t n = 1;
  for (int i = 0; i < nd; ++i) n *= d[i];
  return n;
}

static void contiguous_strides(int nd, const int64_t* d, int64_t* s) {
  int64_t acc = 1;
  for (int i = nd - 1; i >= 0; --i) { s[i] = acc; acc *= d[i]; }
}

RtBuf* rt_buf_new(RT* c, int nd, const int64_t* d, int dev) {
  RtBuf* b = (RtBuf*)calloc(1, sizeof(RtBuf));
  if (!b) { rt_errf(c, "buf OOM"); return NULL; }
  b->device = dev; b->ndim = nd; b->n = (int)rt_numel(nd, d);
  for (int i = 0; i < nd; ++i) b->dims[i] = d[i];
  contiguous_strides(nd, d, b->stride);
  b->host = (float*)malloc(sizeof(float) * (size_t)(b->n ? b->n : 1));
  if (!b->host) { free(b); rt_errf(c, "buf host OOM"); return NULL; }
  b->host_valid = 0;   /* output scratch: no need to upload to device */
  b->ref = 1;
  if (dev == RT_VULKAN && c->vk) { /* device buffer allocated by backend on upload */ }
  return b;
}

void rt_buf_release(RtBuf* b) {
  if (!b) return;
  if (--b->ref > 0) return;
  free(b->host);
  free(b);
}

/* ---------- context ---------- */

RT* rt_create(int device) {
  RT* c = (RT*)calloc(1, sizeof(RT));
  if (!c) return NULL;
  c->device = device;
  c->next_id = 1;
  if (device == RT_VULKAN) rt_vk_init(c);
  return c;
}
void rt_destroy(RT* c) {
  if (!c) return;
  if (c->vk) rt_vk_shutdown(c);
  free(c->nodes);
  free(c);
}
int rt_device(const RT* c) { return c->device; }
const char* rt_last_error(const RT* c) { return c->last_error; }
int rt_dispatch_count(const RT* c) { return c->dispatch_count; }
void rt_reset_dispatch_count(RT* c) { c->dispatch_count = 0; }
const char* rt_version(void) { return RT_VERSION; }

static RTTensor* node_new(RT* c, RTOp op, int nd, const int64_t* d, int nsrc,
                          RTTensor* a, RTTensor* b, RTTensor* k) {
  RTTensor* t = (RTTensor*)calloc(1, sizeof(RTTensor));
  if (!t) { rt_errf(c, "node OOM"); return NULL; }
  t->ctx = c; t->op = op; t->ndim = nd; t->nsrc = nsrc;
  if (nd) { memcpy(t->dims, d, sizeof(int64_t) * nd); contiguous_strides(nd, d, t->stride); }
  t->src[0] = a; t->src[1] = b; t->src[2] = k;
  if (a) a->nconsumer++; if (b) b->nconsumer++; if (k) k->nconsumer++;
  t->id = c->next_id++;
  if (c->nnodes == c->capnodes) {
    c->capnodes = c->capnodes ? c->capnodes * 2 : 64;
    c->nodes = (RTTensor**)realloc(c->nodes, sizeof(RTTensor*) * c->capnodes);
  }
  c->nodes[c->nnodes++] = t;
  if ((a && a->requires_grad) || (b && b->requires_grad) || (k && k->requires_grad))
    t->requires_grad = 1;
  return t;
}

/* ---------- shape inference ---------- */

static void broadcast_shape(RT* c, RTTensor* a, RTTensor* b, int64_t* out, int* nd) {
  int n = a->ndim > b->ndim ? a->ndim : b->ndim;
  if (n > RT_MAXDIMS) { rt_errf(c, "broadcast ndim too large"); *nd = 0; return; }
  for (int i = 0; i < n; ++i) {
    int ia = i - (n - a->ndim), ib = i - (n - b->ndim);
    int64_t da = ia >= 0 ? a->dims[ia] : 1;
    int64_t db = ib >= 0 ? b->dims[ib] : 1;
    if (da != db && da != 1 && db != 1) { rt_errf(c, "broadcast mismatch %lld vs %lld", (long long)da, (long long)db); *nd = 0; return; }
    out[i] = da > db ? da : db;
  }
  *nd = n;
}

/* ---------- frontend ops ---------- */

RTTensor* rt_from_host(RT* c, const float* data, int nd, const int64_t* d, int rg) {
  RTTensor* t = node_new(c, RT_ADD, nd, d, 0, NULL, NULL, NULL);
  t->buf = rt_buf_new(c, nd, d, c->device);
  t->requires_grad = rg;
  int n = t->buf->n;
  if (data && n) {
    memcpy(t->buf->host, data, sizeof(float) * (size_t)n);
  } else if (n) memset(t->buf->host, 0, sizeof(float) * (size_t)n);
  t->buf->host_valid = 1;   /* leaf carries real host data -> upload on first vkbuf */
  return t;
}
RTTensor* rt_zeros(RT* c, int nd, const int64_t* d) { return rt_from_host(c, NULL, nd, d, 0); }

static RTTensor* pointwise(RT* c, RTOp op, RTTensor* a, RTTensor* b) {
  int64_t d[RT_MAXDIMS]; int nd;
  if (b) broadcast_shape(c, a, b, d, &nd);
  else { nd = a->ndim; for (int i=0;i<nd;++i) d[i]=a->dims[i]; }
  if (!nd && (a->ndim || (b && b->ndim))) return NULL;
  RTTensor* t = node_new(c, op, nd, d, b ? 2 : 1, a, b, NULL);
  return t;
}
/* PR #42: fold `matmul + bias` (the nn.Linear pattern) into the GEMM's bias
 * epilogue, which the shader already supports via flag bit 4. Removes a
 * separate add dispatch per linear layer. */
static RTTensor* gemm_bias(RT* c, RTTensor* g, RTTensor* bias) {
  RTTensor* t = node_new(c, RT_MATMUL, g->ndim, g->dims, 3,
                         g->src[0], g->src[1], bias);
  if (!t) return NULL;
  t->axis = g->axis; t->keepdim = g->keepdim;
  return t;
}
RTTensor* rt_add(RT* c, RTTensor* a, RTTensor* b) {
  if (a && !a->buf && a->op == RT_MATMUL && a->nsrc == 2 && b) {
    int64_t N = a->dims[1];
    int bias_ok = (b->ndim == 1 && b->dims[0] == N) ||
                  (b->ndim == 2 && b->dims[0] == 1 && b->dims[1] == N);
    if (bias_ok) return gemm_bias(c, a, b);
  }
  return pointwise(c, RT_ADD, a, b);
}
RTTensor* rt_sub(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_SUB, a, b); }
RTTensor* rt_mul(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_MUL, a, b); }
RTTensor* rt_div(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_DIV, a, b); }
static RTTensor* unary(RT* c, RTOp op, RTTensor* a) { return pointwise(c, op, a, NULL); }
RTTensor* rt_neg(RT* c, RTTensor* a) { return unary(c, RT_NEG, a); }
RTTensor* rt_exp(RT* c, RTTensor* a) { return unary(c, RT_EXP, a); }
RTTensor* rt_log(RT* c, RTTensor* a) { return unary(c, RT_LOG, a); }
RTTensor* rt_sqrt(RT* c, RTTensor* a) { return unary(c, RT_SQRT, a); }
RTTensor* rt_tanh(RT* c, RTTensor* a) { return unary(c, RT_TANH, a); }
RTTensor* rt_sigm(RT* c, RTTensor* a) { return unary(c, RT_SIGM, a); }
/* PR #23: fold an unrealized GEMM + ReLU into one dispatch. Create a distinct
 * RT_GEMM_RELU node (not mutate the GEMM) so other consumers of the GEMM stay
 * correct. */
static RTTensor* gemm_relu(RT* c, RTTensor* a) {
  RTTensor* t = node_new(c, RT_GEMM_RELU, a->ndim, a->dims, a->nsrc,
                         a->src[0], a->src[1], a->src[2]);
  if (!t) return NULL;
  t->axis = a->axis; t->keepdim = a->keepdim;
  return t;
}
RTTensor* rt_relu(RT* c, RTTensor* a) {
  if (a && !a->buf && a->op == RT_MATMUL) return gemm_relu(c, a);
  return unary(c, RT_RELU, a);
}
RTTensor* rt_gelu(RT* c, RTTensor* a) { return unary(c, RT_GELU, a); }
/* PR #2: additional pointwise primitives */
RTTensor* rt_pow(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_POW, a, b); }
RTTensor* rt_maximum(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_MAXIMUM, a, b); }
RTTensor* rt_minimum(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_MINIMUM, a, b); }
RTTensor* rt_abs(RT* c, RTTensor* a) { return unary(c, RT_ABS, a); }
RTTensor* rt_sign(RT* c, RTTensor* a) { return unary(c, RT_SIGN, a); }
RTTensor* rt_gt(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_GT, a, b); }
RTTensor* rt_ge(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_GE, a, b); }
RTTensor* rt_lt(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_LT, a, b); }
RTTensor* rt_le(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_LE, a, b); }
RTTensor* rt_eq(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_EQ, a, b); }
RTTensor* rt_ne(RT* c, RTTensor* a, RTTensor* b) { return pointwise(c, RT_NE, a, b); }
/* Broadcast two shapes without creating a node (pure shape computation). */
static void broadcast_two(const RTTensor* a, const RTTensor* b,
                          int64_t* out, int* nd, RT* c) {
  int n = a->ndim > b->ndim ? a->ndim : b->ndim;
  *nd = n;
  for (int i=0;i<n;++i) {
    int ia=i-(n-a->ndim), ib=i-(n-b->ndim);
    int64_t da=ia>=0?a->dims[ia]:1, db=ib>=0?b->dims[ib]:1;
    if (da!=db && da!=1 && db!=1) { rt_errf(c,"where broadcast mismatch"); *nd=0; return; }
    out[i] = da>db?da:db;
  }
}
RTTensor* rt_where(RT* c, RTTensor* cond, RTTensor* a, RTTensor* b) {
  /* ternary: broadcast all three operands to a common shape */
  int64_t d1[RT_MAXDIMS], d2[RT_MAXDIMS]; int n1, n2;
  broadcast_two(cond, a, d1, &n1, c);
  if (!n1) return NULL;
  /* temp tensors only for shape inference of the 3-way broadcast */
  RTTensor ta = {0};
  ta.ndim=n1; memcpy(ta.dims,d1,sizeof(int64_t)*n1);
  broadcast_two(&ta, b, d2, &n2, c);
  if (!n2) return NULL;
  return node_new(c, RT_WHERE, n2, d2, 3, cond, a, b);
}
/* internal derivative nodes (exposed via backend, fused like pointwise) */
static RTTensor* rt_gt0(RT* c, RTTensor* a) { return unary(c, RT_GT0, a); }
static RTTensor* rt_gelub(RT* c, RTTensor* g, RTTensor* x) { return pointwise(c, RT_GELUB, g, x); }

RTTensor* rt_scalar(RT* c, float value) {
  int64_t d[1] = {1}; float v[1] = {value};
  return rt_from_host(c, v, 1, d, 0);
}

RTTensor* rt_gemm(RT* c, RTTensor* a, RTTensor* b, RTTensor* bias,
                  int trans_a, int trans_b) {
  /* PR #41: fold an unrealized transpose operand into the GEMM's trans flags.
   * X @ W.T() previously dispatched a transpose kernel then a GEMM; the GEMM
   * shader already supports trans_a/trans_b, and grad_wrt already handles the
   * folded flags, so this removes a dispatch with no extra shader work. */
  if (a && !a->buf && a->op == RT_TRANSPOSE && a->nsrc && a->src[0]) {
    a = a->src[0]; trans_a = !trans_a;
  }
  if (b && !b->buf && b->op == RT_TRANSPOSE && b->nsrc && b->src[0]) {
    b = b->src[0]; trans_b = !trans_b;
  }
  if (a->ndim != 2 || b->ndim != 2) { rt_errf(c, "gemm expects 2D"); return NULL; }
  int64_t ka = trans_a ? a->dims[0] : a->dims[1];
  int64_t kb = trans_b ? b->dims[1] : b->dims[0];
  if (ka != kb) { rt_errf(c, "gemm K mismatch %lld vs %lld", (long long)ka, (long long)kb); return NULL; }
  int64_t M = trans_a ? a->dims[1] : a->dims[0];
  int64_t N = trans_b ? b->dims[0] : b->dims[1];
  /* bias may be (N,) or (1,N): identical memory layout, both valid for the
   * shader's bias.v[c] epilogue. nn.Linear uses (1,N). */
  if (bias && !((bias->ndim == 1 && bias->dims[0] == N) ||
                (bias->ndim == 2 && bias->dims[0] == 1 && bias->dims[1] == N))) {
    rt_errf(c, "gemm bias must be shape (N,) or (1,N)"); return NULL;
  }
  int64_t d[2] = { M, N };
  RTTensor* t = node_new(c, RT_MATMUL, 2, d, bias ? 3 : 2, a, b, bias);
  t->axis = trans_a; t->keepdim = trans_b;   /* reuse fields as transpose flags */
  return t;
}
RTTensor* rt_matmul(RT* c, RTTensor* a, RTTensor* b) {
  return rt_gemm(c, a, b, NULL, 0, 0);
}
RTTensor* rt_reduce(RT* c, RTTensor* a, int kind, int axis, int keepdim) {
  if (axis < 0) axis += a->ndim;
  if (axis < 0 || axis >= a->ndim) { rt_errf(c, "reduce axis OOB"); return NULL; }
  int nd = keepdim ? a->ndim : a->ndim - 1;
  if (nd < 1) nd = 1;  /* scalars are 1-element tensors (PyTorch/tinygrad convention) */
  int64_t d[RT_MAXDIMS]; int j = 0;
  for (int i = 0; i < a->ndim; ++i) {
    if (i == axis && keepdim) d[j++] = 1;
    else if (i != axis) d[j++] = a->dims[i];
  }
  if (j == 0) d[j++] = 1;  /* reduced everything: 1-element scalar tensor */
  RTTensor* t = node_new(c, RT_REDUCE, nd, d, 1, a, NULL, NULL);
  t->axis = axis; t->keepdim = keepdim; t->scalar = (float)kind;
  return t;
}
/* PR #24: fused softmax/log_softmax along the last axis (<=256 cols). */
RTTensor* rt_softmax(RT* c, RTTensor* a, RTTensor* bias, int kind) {
  if (!a || a->ndim<1 || a->dims[a->ndim-1]>256) return NULL;
  RTTensor* t=node_new(c,RT_SOFTMAX,a->ndim,a->dims,bias?2:1,a,bias,NULL);
  t->scalar=(float)kind;
  return t;
}
RTTensor* rt_layernorm(RT* c, RTTensor* x, RTTensor* w, RTTensor* b, RTTensor* res, float eps) {
  if (!x || x->ndim<1 || x->dims[x->ndim-1]>256) return NULL;
  RTTensor* t=node_new(c,RT_LAYERNORM,x->ndim,x->dims,3,x,w,b);
  if(res){ t->src[3]=res; t->nsrc=4; res->nconsumer++; }
  t->scalar=eps;
  return t;
}
RTTensor* rt_reduce_sum(RT* c, RTTensor* a, int axis, int keepdim) {
  return rt_reduce(c, a, RT_REDUCE_SUM, axis, keepdim);
}
RTTensor* rt_transpose(RT* c, RTTensor* a) {
  if (a->ndim != 2) { rt_errf(c, "transpose expects 2D"); return NULL; }
  int64_t d[2] = { a->dims[1], a->dims[0] };
  return node_new(c, RT_TRANSPOSE, 2, d, 1, a, NULL, NULL);
}
RTTensor* rt_reshape(RT* c, RTTensor* a, int nd, const int64_t* dims) {
  int64_t n = rt_numel(a->ndim, a->dims), m = rt_numel(nd, dims);
  if (n != m) { rt_errf(c, "reshape element count %lld != %lld", (long long)n, (long long)m); return NULL; }
  return node_new(c, RT_RESHAPE, nd, dims, 1, a, NULL, NULL);
}

void rt_release(RTTensor* t) { /* arena-owned; no-op in 0.2 */ (void)t; }
int rt_ndim(const RTTensor* t) { return t->ndim; }
void rt_shape(const RTTensor* t, int64_t* out) { for (int i=0;i<t->ndim;++i) out[i]=t->dims[i]; }

/* ---------- fusion compiler ---------- */

static int add_input(RtFusion* f, RTTensor* t) {
  for (int i = 0; i < f->ninputs; ++i) if (f->in_tensor[i] == t) return i;
  if (f->ninputs >= RT_MAXIN) return -1;
  int k = f->ninputs++;
  f->in_tensor[k] = t;
  f->in_buf[k] = t->buf;
  /* strides relative to fusion output shape (right aligned, broadcast = 0) */
  for (int j = 0; j < f->ndim; ++j) {
    int ia = j - (f->ndim - t->ndim);
    if (ia >= 0 && ia < t->ndim && !(t->dims[ia] == 1 && f->dims[j] != 1))
      f->in_stride[k][j] = t->buf->stride[ia];
    else f->in_stride[k][j] = 0;
  }
  return k;
}

static int emit(RT* c, RTTensor* t, RtFusion* f);

static int emit_pointwise(RT* c, RTTensor* t, RtFusion* f) {
  RTOp op = t->op;
  RtInstOp io;
  int binary = 0, ternary = 0;
  switch (op) {
    case RT_ADD: io=I_ADD; binary=1; break; case RT_SUB: io=I_SUB; binary=1; break;
    case RT_MUL: io=I_MUL; binary=1; break; case RT_DIV: io=I_DIV; binary=1; break;
    case RT_NEG: io=I_NEG; break; case RT_EXP: io=I_EXP; break;
    case RT_LOG: io=I_LOG; break; case RT_SQRT: io=I_SQRT; break;
    case RT_TANH: io=I_TANH; break; case RT_SIGM: io=I_SIGM; break;
    case RT_RELU: io=I_RELU; break; case RT_GELU: io=I_GELU; break;
    case RT_GT0: io=I_GT0; break; case RT_GELUB: io=I_GELUB; binary=1; break;
    case RT_POW: io=I_POW; binary=1; break;
    case RT_MAXIMUM: io=I_MAXIMUM; binary=1; break; case RT_MINIMUM: io=I_MINIMUM; binary=1; break;
    case RT_ABS: io=I_ABS; break; case RT_SIGN: io=I_SIGN; break;
    case RT_GT: io=I_GT; binary=1; break; case RT_GE: io=I_GE; binary=1; break;
    case RT_LT: io=I_LT; binary=1; break; case RT_LE: io=I_LE; binary=1; break;
    case RT_EQ: io=I_EQ; binary=1; break; case RT_NE: io=I_NE; binary=1; break;
    case RT_WHERE: io=I_WHERE; ternary=1; break;
    default: rt_errf(c, "emit: not pointwise"); return -1;
  }
  if (f->ninst >= RT_MAXINST) { rt_errf(c, "fusion too long"); return -1; }
  if (ternary) {
    int cond = emit(c, t->src[0], f);
    if (cond < 0) return -1;
    int a = emit(c, t->src[1], f);
    if (a < 0) return -1;
    int b = emit(c, t->src[2], f);
    if (b < 0) return -1;
    f->inst[f->ninst].op = io; f->inst[f->ninst].a = cond; f->inst[f->ninst].b = a;
    f->inst[f->ninst].c = *(float*)&b;  /* pack third stack index into float slot */
    return f->ninst++;
  }
  int a = emit(c, t->src[0], f);
  if (a < 0) return -1;
  if (binary) {
    int b = emit(c, t->src[1], f);
    if (b < 0) return -1;
    f->inst[f->ninst].op = io; f->inst[f->ninst].a = a; f->inst[f->ninst].b = b;
  } else {
    f->inst[f->ninst].op = io; f->inst[f->ninst].a = a;
  }
  return f->ninst++;
}

static void realize_node(RT* c, RTTensor* t);
static int is_cut(RTOp op) {
  return op == RT_MATMUL || op == RT_GEMM_RELU || op == RT_REDUCE ||
         op == RT_SOFTMAX || op == RT_LAYERNORM || op == RT_TRANSPOSE || op == RT_RESHAPE;
}

/* Emit expression onto the fused stack; returns the stack slot holding value.
 * Pointwise ancestors are inlined (fusion); cut points are materialised. */
static int emit(RT* c, RTTensor* t, RtFusion* f) {
  if (is_cut(t->op) && !t->buf) realize_node(c, t);
  if (t->buf) {  /* realized leaf (host tensor or cut-point output) */
    int k = add_input(f, t);
    if (k < 0) { rt_errf(c, "too many fusion inputs"); return -1; }
    if (f->ninst >= RT_MAXINST) { rt_errf(c, "fusion too long"); return -1; }
    f->inst[f->ninst].op = I_LOAD; f->inst[f->ninst].a = k;
    return f->ninst++;
  }
  return emit_pointwise(c, t, f);
}

static int RtInstIsBinary(int op) {
  return op==I_ADD||op==I_SUB||op==I_MUL||op==I_DIV||op==I_GELUB||
         op==I_POW||op==I_MAXIMUM||op==I_MINIMUM||
         op==I_GT||op==I_GE||op==I_LT||op==I_LE||op==I_EQ||op==I_NE||op==I_WHERE;
}

/* Translate a fusion's op-program into 25 specialization constants for the
 * specialized straight-line kernel (PR #7). Returns 1 on success, 0 if the
 * program is too complex (caller falls back to the generic interpreter).
 *
 * Generic program: instruction k (index = stack slot it produces). I_LOAD's
 * operand a is the input slot. Other ops reference earlier stack slots.
 * Specialized program: LOADs are stripped; each non-LOAD generic instruction
 * becomes one specialized step s. Operand encoding: 0..7 = input slot,
 * 8..15 = specialized step result. */
static int fusion_to_spec(RtFusion* f) {
  int n = f->ninst;
  if (n < 1 || f->ninputs > 8) return 0;
  /* slot_kind[k]: -1 = I_LOAD, else specialized step index */
  int slot_map[RT_MAXINST];
  for (int k=0;k<n;++k) slot_map[k] = -1;
  int s = 0;
  int spec[25]; for (int i=0;i<25;++i) spec[i]=0;
  for (int k=0;k<n;++k) {
    RtInst in = f->inst[k];
    if (in.op == I_WHERE) return 0;   /* ternary: not supported by specialized path */
    if (in.op == I_LOAD) {
      if (in.a < 0 || in.a >= 8) return 0;
      slot_map[k] = -2 - in.a;   /* encode input slot as -2-slot */
      continue;
    }
    if (s >= 8) return 0;
    /* map operand a (operands are earlier generic slots) */
    if (in.a < 0 || in.a >= n) return 0;
    if (RtInstIsBinary(in.op) && (in.b < 0 || in.b >= n)) return 0;
    int ma = slot_map[in.a];
    int oa;
    if (ma <= -2) oa = -2 - ma;          /* input slot 0..7 */
    else if (ma >= 0) oa = 8 + ma;       /* earlier specialized result */
    else return 0;                        /* forward ref / unmapped */
    if (oa < 0 || oa > 15) return 0;
    int ob = 0;
    if (RtInstIsBinary(in.op)) {
      int mb = slot_map[in.b];
      if (mb <= -2) ob = -2 - mb;
      else if (mb >= 0) ob = 8 + mb;
      else return 0;
      if (ob < 0 || ob > 15) return 0;
    }
    spec[1+s] = in.op; spec[9+s] = oa; spec[17+s] = ob;
    slot_map[k] = s;
    s++;
  }
  if (s == 0) return 0;
  spec[0] = s;
  for (int i=0;i<25;++i) f->sp[i] = spec[i];
  return 1;
}

/* ---------- executors ---------- */

static float gelu_erf(float x) {
  float s = x < 0.f ? -1.f : 1.f, ax = fabsf(x);
  float t = 1.f / (1.f + 0.3275911f * ax);
  float p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
              - 0.284496736f) * t + 0.254829592f) * t;
  return s * (1.f - p * expf(-ax * ax));
}
static float apply1(int op, float a) {
  switch (op) {
    case I_NEG: return -a;
    case I_EXP: return expf(a);
    case I_LOG: return logf(a);
    case I_SQRT: return sqrtf(fmaxf(a, 0.f));
    case I_TANH: return tanhf(a);
    case I_SIGM: return 1.f / (1.f + expf(-a));
    case I_RELU: return a > 0.f ? a : 0.f;
    case I_GELU: { float e = gelu_erf(a * 0.70710678f); return a * 0.5f * (1.f + e); }
    case I_GT0: return a > 0.f ? 1.f : 0.f;
    case I_ABS: return fabsf(a);
    case I_SIGN: return a > 0.f ? 1.f : (a < 0.f ? -1.f : 0.f);
    default: return a;
  }
}
static float apply2(int op, float a, float b) {
  switch (op) {
    case I_ADD: return a + b; case I_SUB: return a - b;
    case I_MUL: return a * b; case I_DIV: return b != 0.f ? a / b : 0.f;
    case I_GELUB: { float e = gelu_erf(b * 0.70710678f);
      float base = 0.5f * (1.f + e);
      float phi = 0.39894228f * expf(-0.5f * b * b);
      return a * (base + b * phi); }  /* a=grad, b=x */
    case I_POW: return powf(a, b);
    case I_MAXIMUM: return a > b ? a : b;
    case I_MINIMUM: return a < b ? a : b;
    case I_GT: return a > b ? 1.f : 0.f;
    case I_GE: return a >= b ? 1.f : 0.f;
    case I_LT: return a < b ? 1.f : 0.f;
    case I_LE: return a <= b ? 1.f : 0.f;
    case I_EQ: return a == b ? 1.f : 0.f;
    case I_NE: return a != b ? 1.f : 0.f;
    default: return a;
  }
}

static void fused_cpu(RtFusion* f, RtBuf* out) {
  int n = out->n, nd = out->ndim;
  float stack[RT_MAXINST];
  for (int e = 0; e < n; ++e) {
    int coord[RT_MAXDIMS]; int rem = e;
    for (int j = nd - 1; j >= 0; --j) { coord[j] = rem % (int)out->dims[j]; rem /= (int)out->dims[j]; }
    int sp = 0;
    for (int i = 0; i < f->ninst; ++i) {
      RtInst in = f->inst[i];
      if (in.op == I_LOAD) {
        int64_t off = 0;
        for (int j = 0; j < nd; ++j) off += coord[j] * f->in_stride[in.a][j];
        stack[sp++] = f->in_buf[in.a]->host[off];
      } else if (in.op == I_WHERE) {
        int cidx; memcpy(&cidx, &in.c, sizeof(int));
        float b = stack[--sp], a = stack[--sp], cond = stack[--sp];
        stack[sp++] = cond != 0.f ? a : b;
      } else if (in.op == I_ADD || in.op == I_SUB || in.op == I_MUL || in.op == I_DIV ||
                 in.op == I_GELUB || in.op == I_POW || in.op == I_MAXIMUM || in.op == I_MINIMUM ||
                 in.op == I_GT || in.op == I_GE || in.op == I_LT || in.op == I_LE ||
                 in.op == I_EQ || in.op == I_NE) {
        float b = stack[--sp], a = stack[--sp];
        stack[sp++] = apply2(in.op, a, b);
      } else {
        float a = stack[--sp];
        stack[sp++] = apply1(in.op, a);
      }
    }
    out->host[e] = stack[sp - 1];
  }
}

static void gemm_cpu(RtBuf* a, RtBuf* b, RtBuf* bias,
                     int ta, int tb, RtBuf* out) {
  int ar=(int)a->dims[0], ac=(int)a->dims[1];
  int br=(int)b->dims[0], bc=(int)b->dims[1];
  int M = ta ? ac : ar, K = ta ? ar : ac;
  int N = tb ? br : bc;
  for (int i=0;i<M;++i) for (int j=0;j<N;++j) {
    float s=0.f;
    for (int k=0;k<K;++k) {
      float av = ta ? a->host[k*M + i] : a->host[i*K + k];
      float bv = tb ? b->host[j*K + k] : b->host[k*N + j];
      s += av*bv;
    }
    if (bias) s += bias->host[j];
    out->host[i*N+j]=s;
  }
}
static void reduce_cpu(RtBuf* in, int kind, int axis, RtBuf* out) {
  int outer=1, stride=(int)in->dims[axis], inner=1;
  for (int i=0;i<axis;++i) outer*=(int)in->dims[i];
  for (int i=axis+1;i<in->ndim;++i) inner*=(int)in->dims[i];
  for (int o=0;o<outer;++o) for (int t=0;t<inner;++t) {
    float s = (kind==RT_REDUCE_MAX) ? -INFINITY : (kind==RT_REDUCE_MIN ? INFINITY : 0.f);
    for (int k=0;k<stride;++k) {
      float v = in->host[(o*stride+k)*inner+t];
      if (kind==RT_REDUCE_MAX) s = fmaxf(s, v);
      else if (kind==RT_REDUCE_MIN) s = fminf(s, v);
      else s += v;
    }
    if (kind==RT_REDUCE_MEAN) s /= (float)stride;
    out->host[o*inner+t]=s;
  }
}

/* ---------- scheduler ---------- */

static void realize_node(RT* c, RTTensor* t);

static void realize_fused(RT* c, RTTensor* t) {
  /* A single fusion can hold at most RT_MAXIN distinct leaf inputs. Deep
   * composition trees (e.g. isnan/isinf built from several scalar constants)
   * can exceed that. If emit fails, materialize the direct children into
   * buffers and retry: each child then becomes a single leaf input. Never
   * leave t->buf NULL — that silently read as zeros (fixed here). */
  for (int attempt=0; attempt<4; ++attempt) {
    RtFusion f; memset(&f, 0, sizeof(f));
    f.ndim = t->ndim;
    for (int i=0;i<t->ndim;++i) f.dims[i]=t->dims[i];
    int slot = emit(c, t, &f);
    if (slot < 0) {
      if (attempt == 3) { /* give up: materialize an explicit zero buffer */
        RtBuf* z = rt_buf_new(c, t->ndim, t->dims, c->device);
        if (z) { memset(z->host, 0, sizeof(float)*(size_t)(z->n?z->n:1)); t->buf = z; }
        return;
      }
      for (int i=0;i<t->nsrc;++i) if (t->src[i]) realize_node(c, t->src[i]);
      continue;
    }
    if (f.ninst == 0) { rt_errf(c, "empty fusion"); return; }
    f.sp_ok = fusion_to_spec(&f);
    RtBuf* out = rt_buf_new(c, t->ndim, t->dims, c->device);
    if (!out) return;
    if (c->device == RT_VULKAN && c->vk) rt_vk_fused(c, &f, out);  /* records if capturing */
    else fused_cpu(&f, out);
    if (c->capture && c->capture_nout < RT_MAXBUF) c->capture_outputs[c->capture_nout++] = out;
    c->dispatch_count++;
    t->buf = out;
    return;
  }
}

static void realize_node(RT* c, RTTensor* t) {
  if (t->buf) return;
  if (!is_cut(t->op)) { realize_fused(c, t); return; }  /* emit() materialises cut ancestors */
  for (int i=0;i<t->nsrc;++i) if (t->src[i]) realize_node(c, t->src[i]);
  if (t->op == RT_MATMUL || t->op == RT_GEMM_RELU) {
    RtBuf* out = rt_buf_new(c, 2, t->dims, c->device);
    RtBuf* bias = t->nsrc > 2 && t->src[2] ? t->src[2]->buf : NULL;
    int ta = t->axis, tb = t->keepdim;
    int relu = t->op == RT_GEMM_RELU;
    if (c->device == RT_VULKAN && c->vk)
      rt_vk_gemm(c, t->src[0]->buf, t->src[1]->buf, bias, ta, tb, relu, out);
    else gemm_cpu(t->src[0]->buf, t->src[1]->buf, bias, ta, tb, out);
    if (relu && c->device != RT_VULKAN) {   /* CPU: apply ReLU epilogue */
      for (int i=0;i<out->n;++i) if (out->host[i]<0) out->host[i]=0;
    }
    if (c->capture && c->capture_nout < RT_MAXBUF) c->capture_outputs[c->capture_nout++] = out;
    c->dispatch_count++; t->buf = out; return;
  }
  if (t->op == RT_REDUCE) {
    RtBuf* out = rt_buf_new(c, t->ndim, t->dims, c->device);
    int kind = (int)t->scalar;
    if (c->device == RT_VULKAN && c->vk)
      rt_vk_reduce(c, t->src[0]->buf, kind, t->axis, out);
    else reduce_cpu(t->src[0]->buf, kind, t->axis, out);
    if (c->capture && c->capture_nout < RT_MAXBUF) c->capture_outputs[c->capture_nout++] = out;
    c->dispatch_count++; t->buf = out; return;
  }
  if (t->op == RT_RESHAPE) {
    RtBuf* in = t->src[0]->buf;
    if (in && in->dirty && c->vk) rt_vk_download(c, in);   /* host copy needs fresh data */
    RtBuf* out = rt_buf_new(c, t->ndim, t->dims, c->device);
    memcpy(out->host, in->host, sizeof(float) * (size_t)out->n);
    out->host_valid = 1;   /* host-side copy -> must upload to device on first vkbuf */
    c->dispatch_count++; t->buf = out; return;
  }
  if (t->op == RT_TRANSPOSE) {
    /* PR #35: device-side transpose so it participates in capture/replay. A
     * host-side memcpy reads stale b->host when the input lives in a slot. */
    RtBuf* in = t->src[0]->buf;
    RtBuf* out = rt_buf_new(c, 2, t->dims, c->device);
    if (c->device==RT_VULKAN && c->vk) rt_vk_transpose(c,in,out);
    else { int M=(int)in->dims[0], N=(int)in->dims[1];
      for (int i=0;i<M;++i) for (int j=0;j<N;++j) out->host[j*M+i]=in->host[i*N+j]; }
    if (c->capture && c->capture_nout<RT_MAXBUF) c->capture_outputs[c->capture_nout++]=out;
    c->dispatch_count++; t->buf = out; return;
  }
  if (t->op == RT_LAYERNORM) {
    RtBuf* x=t->src[0]->buf, *w=t->src[1]->buf, *b=t->src[2]->buf;
    RtBuf* res=t->nsrc>3?t->src[3]->buf:NULL;
    RtBuf* out=rt_buf_new(c,t->ndim,t->dims,c->device);
    if (c->device==RT_VULKAN && c->vk) rt_vk_layernorm(c,x,w,b,res,out,t->scalar);
    else { int cols=(int)x->dims[x->ndim-1], rows=x->n/cols; float eps=t->scalar;
      for(int r=0;r<rows;++r){ float m=0,vv=0;
        for(int j=0;j<cols;++j) m+=x->host[r*cols+j]+(res?res->host[r*cols+j]:0.f); m/=cols;
        for(int j=0;j<cols;++j){ float d=x->host[r*cols+j]+(res?res->host[r*cols+j]:0.f)-m; vv+=d*d; } vv/=cols;
        float sc=1.0f/sqrtf(vv+eps);
        for(int j=0;j<cols;++j) out->host[r*cols+j]=(x->host[r*cols+j]+(res?res->host[r*cols+j]:0.f)-m)*sc*w->host[j]+b->host[j];
      }
    }
    c->dispatch_count++; t->buf=out; return;
  }
  if (t->op == RT_SOFTMAX) {
    RtBuf* in=t->src[0]->buf;
    RtBuf* bias=t->nsrc>1?t->src[1]->buf:NULL;
    RtBuf* out=rt_buf_new(c,t->ndim,t->dims,c->device);
    int kind=(int)t->scalar;
    if (c->device==RT_VULKAN && c->vk) rt_vk_softmax(c,in,bias,out,kind);
    else { /* CPU fallback: stable softmax row by row */
      int cols=(int)in->dims[in->ndim-1], rows=in->n/cols;
      for(int r=0;r<rows;++r){ float mx=-INFINITY;
        for(int j=0;j<cols;++j) mx=fmaxf(mx,in->host[r*cols+j]+(bias?bias->host[j]:0.f));
        float s=0.f; for(int j=0;j<cols;++j){ float e=expf(in->host[r*cols+j]+(bias?bias->host[j]:0.f)-mx);
          out->host[r*cols+j]=e; s+=e; }
        for(int j=0;j<cols;++j){ float v=out->host[r*cols+j]/s;
          out->host[r*cols+j]=(kind==1)?logf(v):v; }
      }
    }
    c->dispatch_count++; t->buf=out; return;
  }
  realize_fused(c, t);
}

void rt_realize(RT* c, RTTensor* t) { realize_node(c, t); }

void rt_to_host(RTTensor* t, float* out) {
  if (!t->buf) rt_realize(t->ctx, t);
  if (!t->buf) return;
  if (t->buf->dirty && t->ctx->vk) {
    rt_vk_download_to(t->ctx, t->buf, out);   /* direct staging->user, no cache copy */
    return;
  }
  memcpy(out, t->buf->host, sizeof(float) * (size_t)t->buf->n);
}

/* ---------- autograd ---------- */

static RTTensor* scalar(RT* c, float v) {
  int64_t d[1] = {1}; float x[1] = {v};
  return rt_from_host(c, x, 1, d, 0);
}
/* Reduce grad g to parent shape (broadcast axes collapse). */
static RTTensor* unbroadcast(RT* c, RTTensor* g, RTTensor* parent) {
  RTTensor* cur = g;
  while (cur->ndim > parent->ndim) cur = rt_reduce_sum(c, cur, 0, 0);
  for (int ax = 0; ax < parent->ndim; ++ax) {
    if (parent->dims[ax] == 1 && cur->dims[ax] != 1)
      cur = rt_reduce_sum(c, cur, ax, 1);
  }
  return cur;
}
static void accum(RT* c, RTTensor* parent, RTTensor* g) {
  if (!parent->requires_grad) return;
  RTTensor* ug = unbroadcast(c, g, parent);
  parent->grad = parent->grad ? rt_add(c, parent->grad, ug) : ug;
}

/* Gradient of node t w.r.t. source index s, given upstream gradient g. */
static RTTensor* grad_wrt(RT* c, RTTensor* t, int s, RTTensor* g) {
  RTTensor* a = t->nsrc > 0 ? t->src[0] : NULL;
  RTTensor* b = t->nsrc > 1 ? t->src[1] : NULL;
  switch (t->op) {
    case RT_ADD: return g;
    case RT_SUB: return s == 0 ? g : rt_neg(c, g);
    case RT_MUL: return s == 0 ? (b ? rt_mul(c, g, b) : g) : rt_mul(c, g, a);
    case RT_DIV: {
      if (s == 0) return b ? rt_div(c, g, b) : g;
      RTTensor* num = a;
      return rt_neg(c, rt_div(c, rt_mul(c, g, num), rt_mul(c, b, b))); }
    case RT_NEG: return rt_neg(c, g);
    case RT_EXP: return rt_mul(c, g, t);
    case RT_LOG: return rt_div(c, g, a);
    case RT_SQRT: { RTTensor* two=scalar(c,2.f); return rt_div(c, g, rt_mul(c, two, t)); }
    case RT_TANH: { RTTensor* one=scalar(c,1.f); RTTensor* o2=rt_mul(c,t,t);
      return rt_mul(c, g, rt_sub(c, one, o2)); }
    case RT_SIGM: { RTTensor* one=scalar(c,1.f);
      return rt_mul(c, g, rt_mul(c, t, rt_sub(c, one, t))); }
    case RT_RELU: return rt_mul(c, g, rt_gt0(c, a));
    case RT_GELU: return rt_gelub(c, g, a);
    case RT_ABS: return rt_mul(c, g, rt_sign(c, a));
    case RT_SIGN: return scalar(c, 0.f);
    case RT_POW: {  /* d(a^b)/da = b*a^(b-1), d/db = a^b*log(a) */
      if (s == 0) { RTTensor* b1 = rt_sub(c, b, scalar(c,1.f));
        return rt_mul(c, g, rt_mul(c, b, rt_pow(c, a, b1))); }
      return rt_mul(c, g, rt_mul(c, t, rt_log(c, a))); }
    case RT_MAXIMUM: return rt_mul(c, g, rt_ge(c, a, b));
    case RT_MINIMUM: return rt_mul(c, g, rt_le(c, a, b));
    case RT_GT: case RT_GE: case RT_LT: case RT_LE: case RT_EQ: case RT_NE:
      return scalar(c, 0.f);
    case RT_WHERE: return s == 1 ? rt_mul(c, g, t->src[0])
                                 : (s == 2 ? rt_mul(c, g, rt_sub(c, scalar(c,1.f), t->src[0])) : scalar(c,0.f));
    case RT_MATMUL:
    case RT_GEMM_RELU: {
      /* Y = relu(opA(A) @ opB(B) + bias); for GEMM_RELU mask g by gt0(Y). */
      if (t->op == RT_GEMM_RELU) g = rt_mul(c, g, rt_gt0(c, t));
      /* Y = opA(A) @ opB(B), opX = transpose iff flag set.
       * dY/dA' = g @ B'^T ; dY/dB' = A'^T @ g ; then un-transpose. */
      int ta = t->axis, tb = t->keepdim;
      RTTensor* A = t->src[0], *B = t->src[1];
      if (s == 0) {
        RTTensor* BpT = tb ? B : rt_transpose(c, B);   /* opB(B)^T */
        RTTensor* dAp = rt_matmul(c, g, BpT);          /* dY/dA' */
        return ta ? rt_transpose(c, dAp) : dAp;
      }
      if (s == 1) {
        RTTensor* ApT = ta ? A : rt_transpose(c, A);   /* opA(A)^T */
        RTTensor* dBp = rt_matmul(c, ApT, g);          /* dY/dB' */
        return tb ? rt_transpose(c, dBp) : dBp;
      }
      /* s==2: bias. dY/dbias[j] = sum_i g[i,j] -> reduce over axis 0 with
       * keepdim so unbroadcast() handles (1,N) biases correctly. */
      return rt_reduce(c, g, RT_REDUCE_SUM, 0, 1);
    }
    case RT_LAYERNORM: {
      /* y = xhat*s*w + b, xhat=(x-mean)/sqrt(var+eps), s=rsqrt(var+eps)
       * dx = (1/n)*s*(n*g*w - sum(g*w) - xhat*sum(g*w*xhat))
       * gw = sum_rows(g*xhat), gb = sum_rows(g) */
      RTTensor* x=t->src[0], *w=t->src[1];
      if(t->nsrc>3 && t->src[3]) x=rt_add(c,x,t->src[3]);  /* u = x + residual */
      /* w may carry a leading 1 dim ((1,cols)); the normalised axis is always
       * the last axis of the widest operand, not of x (fixes 1-D LN inputs). */
      int wnd = w->ndim > x->ndim ? w->ndim : x->ndim;
      int ax=wnd-1, cols=(int)x->dims[x->ndim-1];
      RTTensor* mean=rt_reduce(c,x,RT_REDUCE_MEAN,x->ndim-1,1);
      RTTensor* d=rt_sub(c,x,mean);
      RTTensor* var=rt_reduce(c,rt_mul(c,d,d),RT_REDUCE_MEAN,x->ndim-1,1);
      RTTensor* inv=rt_pow(c,rt_add(c,var,scalar(c,t->scalar)),scalar(c,-0.5f));
      RTTensor* xhat=rt_mul(c,d,inv);
      /* dw/db: sum over all leading axes, keep the last (feature) axis.
       * A 1-D input has no leading axes -> no reduction needed. */
      if(s==1){ RTTensor* gw=rt_mul(c,g,xhat);
        for(int lead=x->ndim-2;lead>=0;--lead) gw=rt_reduce(c,gw,RT_REDUCE_SUM,lead,1);
        return rt_reshape(c,gw,w->ndim,w->dims); }
      if(s==2){ RTTensor* gb=g; RTTensor* bias=t->src[2];
        for(int lead=x->ndim-2;lead>=0;--lead) gb=rt_reduce(c,gb,RT_REDUCE_SUM,lead,1);
        return rt_reshape(c,gb,bias->ndim,bias->dims); }
      RTTensor* dxh=rt_mul(c,g,w);
      RTTensor* s1=rt_reduce(c,dxh,RT_REDUCE_SUM,ax,1);
      RTTensor* s2=rt_reduce(c,rt_mul(c,dxh,xhat),RT_REDUCE_SUM,ax,1);
      /* dx = (n*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat)) * inv / n */
      RTTensor* core=rt_sub(c,rt_sub(c,rt_mul(c,dxh,scalar(c,(float)cols)),s1),
                            rt_mul(c,xhat,s2));
      return rt_mul(c,rt_mul(c,core,inv),scalar(c,1.0f/(float)cols));
    }
    case RT_SOFTMAX: {
      /* softmax: dx = y*(g - sum(g*y)); log_softmax: dx = g - exp(y)*sum(g) */
      int ax=t->ndim-1, kind=(int)t->scalar;
      RTTensor* gy = (kind==0) ? rt_mul(c,g,t) : rt_mul(c,g,rt_exp(c,t));
      if (s==1) return rt_reduce(c,gy,RT_REDUCE_SUM,0,1);  /* dbias */
      if (kind==0) {
        RTTensor* sumg=rt_reduce(c,gy,RT_REDUCE_SUM,ax,1);
        return rt_mul(c,t,rt_sub(c,g,sumg));
      } else {
        RTTensor* sumg=rt_reduce(c,g,RT_REDUCE_SUM,ax,1);
        return rt_sub(c,g,rt_mul(c,rt_exp(c,t),sumg));
      }
    }
    case RT_REDUCE: {
      int kind = (int)t->scalar;
      int64_t n = rt_numel(a->ndim, a->dims);
      float* ones = (float*)malloc(sizeof(float) * (size_t)n);
      for (int64_t i=0;i<n;++i) ones[i] = 1.f;
      RTTensor* ot = rt_from_host(c, ones, a->ndim, a->dims, 0);
      free(ones);
      /* broadcast g to input shape; t is the reduce output (scalar for these tests) */
      RTTensor* gfull = rt_mul(c, g, ot);
      if (kind == RT_REDUCE_SUM) return gfull;
      if (kind == RT_REDUCE_MEAN) {
        int64_t count = rt_numel(a->ndim, a->dims) / rt_numel(t->ndim, t->dims);
        return rt_mul(c, gfull, scalar(c, 1.f/(float)count)); }
      if (kind == RT_REDUCE_MAX || kind == RT_REDUCE_MIN) {
        /* mask = (a == broadcast(t)); grad = gfull * mask */
        RTTensor* tbcast = rt_mul(c, t, ot);              /* scalar t broadcast to input shape */
        RTTensor* mask = rt_eq(c, a, tbcast);
        return rt_mul(c, gfull, mask); }
      return gfull; }
    case RT_TRANSPOSE: return rt_transpose(c, g);
    case RT_RESHAPE: return rt_reshape(c, g, a->ndim, a->dims);
    case RT_GT0: case RT_GELUB: return scalar(c, 0.f);
    default: return scalar(c, 0.f);
  }
}

static void topo_collect(RT* c, RTTensor* root, RTTensor** out, int* n) {
  /* iterative postorder DFS with a visited marker array */
  int cap = c->nnodes + 1;
  unsigned char* seen = (unsigned char*)calloc((size_t)cap, 1);
  RTTensor** st = (RTTensor**)malloc(sizeof(RTTensor*) * (size_t)cap * 4);
  int sp = 0, cnt = 0;
  st[sp++] = root;
  while (sp) {
    RTTensor* nd = st[--sp];
    int idx = (int)(nd->id - 1);
    if (idx < 0 || idx >= cap || seen[idx]) continue;
    int allseen = 1;
    for (int i=0;i<nd->nsrc;++i) if (nd->src[i]) {
      int ci = (int)(nd->src[i]->id - 1);
      if (ci >= 0 && ci < cap && !seen[ci]) { st[sp++]=nd; st[sp++]=nd->src[i]; allseen=0; break; }
    }
    if (allseen) { seen[idx]=1; out[cnt++]=nd; }
  }
  free(st); free(seen); *n = cnt;
}

void rt_backward(RT* c, RTTensor* loss) {
  rt_realize(c, loss);
  RTTensor** topo = (RTTensor**)malloc(sizeof(RTTensor*) * (size_t)(c->nnodes + 1));
  int nt = 0; topo_collect(c, loss, topo, &nt);
  if (loss->grad == NULL) {
    int64_t d[1] = {1}; float one[1] = {1.f};
    loss->grad = rt_from_host(c, one, 1, d, 0);
  }
  for (int i = nt - 1; i >= 0; --i) {
    RTTensor* n = topo[i];
    if (n->grad == NULL) continue;
    rt_realize(c, n->grad);
    for (int s = 0; s < n->nsrc; ++s) {
      RTTensor* p = n->src[s];
      if (!p || !p->requires_grad) continue;
      RTTensor* g = grad_wrt(c, n, s, n->grad);
      accum(c, p, g);
    }
  }
  /* PR #35: realise every requires_grad leaf's final grad inside the capture
   * window so its dispatch writes to a capture slot, not a stale eager buffer. */
  for (int i=0;i<c->nnodes;++i) {
    RTTensor* n=c->nodes[i];
    if (n->requires_grad && n->grad) rt_realize(c,n->grad);
  }
  free(topo);
}
RTTensor* rt_grad(const RTTensor* t) { return t->grad; }

const float* rt_read_ptr(RT* c, RTTensor* t, int slot) {
  if (!t) return NULL;
  if (!t->buf) rt_realize(c, t);
  if (!t->buf) return NULL;
  if (c->vk) return rt_vk_read_ptr(c, t->buf, slot);
  return (const float*)t->buf->host;
}

void rt_zero_grad(RTTensor* t) { if (t) t->grad=NULL; }
void rt_stream_begin(RT* c, int idx) { if (c && c->vk) rt_vk_stream_begin(c,idx); }
void rt_stream_end(RT* c, int idx) { if (c && c->vk) rt_vk_stream_end(c,idx); }
void rt_stream_submit(RT* c, int n) { if (c && c->vk) rt_vk_stream_submit(c,n); }

float* rt_tensor_map(RT* c, RTTensor* t) {
  if (!c || !t) return NULL;
  if (!t->buf) rt_realize(c,t);
  if (!t->buf) return NULL;
  if (c->vk) return rt_vk_map_input(c,t->buf);
  return (float*)t->buf->host;
}
void rt_tensor_flush(RT* c, RTTensor* t) {
  if (c && t && t->buf && c->vk) rt_vk_flush_input(c,t->buf);
}

void rt_sgd_update(RT* c, RTTensor* w, RTTensor* grad, float lr, float wd) {
  if (!c || !w || !grad) return;
  if (!w->buf) rt_realize(c,w);
  if (!grad->buf) rt_realize(c,grad);
  if (c->vk) rt_vk_sgd_update(c,w->buf,grad->buf,lr,wd);
  else { for(int i=0;i<w->buf->n;++i) w->buf->host[i]-=lr*(grad->buf->host[i]+wd*w->buf->host[i]); }
}

double rt_device_time_ms(RT* c) {
  if (!c) return 0.0;
  if (c->vk) rt_vk_flush_timestamps(c);
  return c->last_gpu_ms;
}
double rt_copy_time_ms(RT* c) {
  if (!c) return 0.0;
  if (c->vk) rt_vk_flush_timestamps(c);
  return c->copy_gpu_ms;
}

/* ---------- capture / replay ---------- */

void rt_upload(RT* c, RTTensor* t, const float* data) {
  (void)c;
  if (!t->buf) rt_realize(t->ctx, t);
  if (!t->buf) return;
  if (t->buf->device == RT_VULKAN && t->ctx->vk) { rt_vk_upload(t->ctx, t->buf, data); return; }
  memcpy(t->buf->host, data, sizeof(float) * (size_t)t->buf->n);
}

void rt_capture_begin(RT* c) { c->capture = 1; c->capture_n = 0; c->capture_nsrc = 0;
  c->capture_nout = 0;
  if (c->vk) rt_vk_capture_begin(c); }
void rt_capture_end(RT* c) { c->capture = 0;
  if (c->vk) rt_vk_capture_end(c); }
void rt_replay(RT* c) {
  if (c->vk) { rt_vk_replay(c); return; }  /* outputs copied back by rt_vk_replay */
  /* CPU replay: re-run captured fused programs against current src bufs */
  for (int i=0;i<c->capture_n;++i) {
    RtFusion* f = &c->capture_fusion[i];
    RtBuf out; memset(&out,0,sizeof(out));
    out.device=RT_CPU; out.ndim=f->ndim; out.n=(int)rt_numel(f->ndim,f->dims);
    for (int j=0;j<f->ndim;++j) out.dims[j]=f->dims[j];
    contiguous_strides(f->ndim, f->dims, out.stride);
    out.host=(float*)malloc(sizeof(float)*(size_t)(out.n?out.n:1));
    /* refresh input buf pointers from current tensors */
    for (int k=0;k<f->ninputs;++k) f->in_buf[k]=f->in_tensor[k]->buf;
    fused_cpu(f, &out);
    /* write back into the captured output tensor buffer */
    /* (stored as the last captured src for this fusion in order) */
    free(out.host);
    c->dispatch_count++;
  }
}

void rt_replay_submit_only(RT* c) {
  if (!c->vk) return;
  rt_vk_submit_only(c);
  /* mark captured outputs dirty; downloaded lazily by rt_to_host */
  for (int i=0;i<c->capture_nout;++i) c->capture_outputs[i]->dirty = 1;
}
void rt_replay_submit_async(RT* c) {
  if (!c->vk) return;
  rt_vk_submit_async(c);
  for (int i=0;i<c->capture_nout;++i) c->capture_outputs[i]->dirty = 1;
}
void rt_replay_n(RT* c, int n) {
  if (!c->vk) return;
  rt_vk_replay_n(c,n);
  for (int i=0;i<c->capture_nout;++i) c->capture_outputs[i]->dirty = 1;
}
