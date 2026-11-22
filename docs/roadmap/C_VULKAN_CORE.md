# C + Vulkan core — complete function map (v0.2 roadmap)

This is the low-level implementation inventory for the C11 + Vulkan core. It
maps **every function the core must expose**, grouped by module, with a one-line
contract, the kernel(s) it lowers to, and which numeric-parity case verifies it.
It is a planning document — no code exists on this branch. Design rationale is
in `docs/research/FRAMEWORKS.md`; the line budget is in `../LINE_BUDGET.md`.

Conventions:
- fp32 is the only compute dtype in v0.2.
- Every op is **lazy**: the C function builds a node; nothing executes until
  `bt_realize`. Autograd backward functions are marked *(bwd)*.
- "fused" = folded into the one op-program kernel (`fused.comp`); "cut" = a
  materializing kernel (GEMM / reduce). This is tinygrad's "one CALL = one
  kernel" rule.
- Tolerance classes: EXACT (bit-identical), APPROX 1e-5 (transcendental/matmul),
  MODEL 1e-3 (loss). See `../FEATURE_PARITY_MATRIX.md`.

---

## Module 1 — `core/` : opaque types, device, allocator, shape, error

| Function | Contract | Kernel | Verified by |
|---|---|---|---|
| `bt_init(flags)` | one-time init; selects device, Vulkan if enabled | — | CTest `unit_core` |
| `bt_shutdown()` | release device, arenas, caches | — | CTest |
| `bt_device_create(kind)` | acquire CPU or Vulkan device handle | — | smoke |
| `bt_device_release(d)` | release device | — | smoke |
| `bt_shape_from_dims(dims,n)` | build immutable shape + contiguous strides | — | unit |
| `bt_shape_numel(s)` | element count | — | unit |
| `bt_broadcast_shapes(a,b,out)` | right-aligned broadcast (size 1 expands); error on mismatch | — | add_broadcast cases |
| `bt_arena_create(bytes, dev)` | one big `DEVICE_LOCAL(+HOST_VISIBLE)` allocation, bump cursor | — | unit |
| `bt_arena_alloc(a,bytes,align)` | aligned suballocation; reuse on graph teardown | — | perf |
| `bt_arena_reset(a)` | bulk free (graph teardown) | — | perf |
| `bt_buffer_create(dev,bytes)` | wrap an arena slice as a buffer | — | unit |
| `bt_buffer_upload(buf,host,n)` | stage + copy to device w/ barrier | copy | parity setup |
| `bt_buffer_download(buf,host,n)` | readback w/ barrier | copy | all parity cases |
| `bt_set_error(fmt,...)` / `bt_last_error()` | thread-local error state | — | error cases |
| `bt_version()` | version string | — | environment report |

## Module 2 — `graph/` : the one IR node, lazy frontend, views

| Function | Contract | Lowers to | Notes |
|---|---|---|---|
| `bt_node(op,src,n,dtype,arg)` | hash-consed node constructor (pointer equality = structural) | — | tinygrad UOp |
| `bt_node_shape(n)` | infer shape (broadcast for binary, permute for views) | — | — |
| `bt_scalar(v)` | const node | fused CONST | backward constants |
| `bt_from_host(ptr,shape,dev)` | leaf node with realized buffer | — | input |
| `bt_zeros(shape)` | zero leaf | — | init |
| `bt_add/sub/mul/div(a,b)` | binary pointwise (broadcast) | fused | (bwd): g / −g / g·b,g·a / quotient rule |
| `bt_neg(a)` | unary | fused NEG | (bwd) −g |
| `bt_exp/log/sqrt(a)` | unary transcendental | fused EXP/LOG/SQRT | (bwd) g·out / g/x / g/(2·out) |
| `bt_tanh/sigmoid/relu/gelu(a)` | unary activation | fused TANH/SIGM/RELU/GELU | (bwd) analytic; gelu uses erf+GELUB |
| `bt_gt0(a)` / `bt_gelub(g,x)` | derivative primitives | fused | relu/gelu backward |
| `bt_matmul(a,b)` | 2D (M,K)·(K,N) | **cut → gemm** | (bwd) g·Bᵀ, Aᵀ·g |
| `bt_reduce_sum(a,axis,keepdim)` | sum along axis | **cut → reduce** | (bwd) broadcast g |
| `bt_transpose(a)` | 2D view (strides, no copy) | view | (bwd) transpose g |
| `bt_reshape(a,shape)` | contiguous view (no copy if compatible) | view | — |
| `bt_realize(n)` | schedule + execute the graph rooted at n | scheduler | the only execution trigger |
| `bt_to_host(n,out)` | realize + download | — | parity readback |

## Module 3 — `sched/` : fusion scheduler

| Function | Contract | Notes |
|---|---|---|
| `bt_topo(n,out)` | topological order of reachable nodes | iterative DFS |
| `bt_build_fusion(root)` | absorb maximal pointwise chains into one op-program; cut at matmul/reduce | one program = one dispatch |
| `bt_plan_buffers(graph)` | assign arena slots; views are strides not copies | memory planner |
| `bt_compile_program(fusion)` | encode op-code program + operand descriptors into push/SSBO layout | reuse by hash key |
| `bt_dispatch_count()` | # kernels executed since reset (fusion proof) | must be 1 for an N-op chain |

## Module 4 — `codegen/` : three parameterized kernels

| Kernel file | Specialization constants | Purpose |
|---|---|---|
| `fused.comp` | none (op-program in SSBO) | ALL elementwise chains, reductions-free |
| `gemm.comp` | BM, BN, BK (tile sizes) | tiled fp32 GEMM; autotune over a few tuples |
| `reduce.comp` | axis, keepdim | generic sum reduction |

## Module 5 — `exec/` : Vulkan device + capture/replay

| Function | Contract | Vulkan primitives |
|---|---|---|
| `bt_vk_init()` | instance (1.3), pick discrete GPU, compute queue | `vkCreateInstance`, `vkEnumeratePhysicalDevices` |
| `bt_vk_features()` | enable buffer device address, sync2, timeline, (opt-in shader object / coop matrix) | `VkPhysicalDevice*Features` chains |
| `bt_vk_pipeline(spv,spec)` | compile one SPIR-V + spec constants into a pipeline, cache by key | `vkCreateComputePipelines`, `VkPipelineCache` |
| `bt_vk_bindless_set()` | one descriptor set: SSBO of device addresses + lengths | `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`, BDA |
| `bt_vk_dispatch(prog,shape)` | push constants (shape/op offset) + `vkCmdDispatch` | `vkCmdPushConstants` |
| `bt_vk_barrier()` | compute→compute memory dependency | `vkCmdPipelineBarrier2` (`VkMemoryBarrier2`) |
| `bt_vk_flush()` | submit recorded work, signal timeline, no wait | `vkQueueSubmit2` |
| `bt_vk_wait()` | wait timeline value; collect readback | `vkWaitSemaphores` / `VkFence` |
| `bt_capture_begin()/end()` | record all dispatches into one primary command buffer | `vkBeginCommandBuffer` once |
| `bt_replay()` | resubmit prebuilt CB with fresh timeline value; zero per-step driver work | `vkQueueSubmit2` + timeline (the tinygrad trick) |
| `bt_cpu_*` | reference backend using the same scheduler | correctness oracle + CI (no GPU) |

## Module 6 — `ops/` composition layer (public convenience, tiny)

| Function | Composition | Parity case |
|---|---|---|
| `bt_softmax(x)` | exp(x − reduce_max) / reduce_sum | softmax cases |
| `bt_log_softmax(x)` | x − log(sum exp x) | ce_loss |
| `bt_cross_entropy(logits,onehot)` | log_softmax + reduce + neg | GPT-2 loss 0.0 |
| `bt_mse_loss(p,t)` | reduce(mean((p−t)²)) | MLP |
| `bt_layer_norm(x,W,b,eps)` | reduce mean/var + fused affine | LN ≤5.4e-7 |
| `bt_sgd_step(p,g,lr)` / `bt_adamw_step(...)` | fused update kernel | W 1.5e-8 |

---

## End-to-end dependency order (implementation sequence)

1. `core/` (device, arena, shape, error) → CTest green.
2. `graph/` node + the 6 leaf/binary/unary ops → `bt_realize` on CPU backend.
3. `sched/` fusion + `fused.comp` → "N-op chain = 1 dispatch" proof.
4. `gemm.comp` + `reduce.comp` → matmul/reduce parity.
5. bwd rules → gradcheck parity.
6. `ops/` composition → softmax/LN/CE/losses parity.
7. Vulkan `exec/` bindless + capture/replay → speed benchmarks.
8. AdamW + model parity scripts → full roadmap parity matrix green.

## Line-budget cross-check

Per `../LINE_BUDGET.md`: core subtotal target 2850 (hard cap 4096). The module
order above lets us stop and measure after each step; `tools/count_lines.py`
must stay under budget at every commit. Each new function belongs to exactly
one module row above; no per-op C dispatch functions (that is what bloated the
legacy core to 5710 lines).
