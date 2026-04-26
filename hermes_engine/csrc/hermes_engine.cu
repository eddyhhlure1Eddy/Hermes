#include <torch/extension.h>
#include <pybind11/stl.h>          // std::vector<Tensor>, std::optional bindings
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <mutex>
#include <optional>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(),  #x " must be cuda")
#define CHECK_CTG(x)  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == torch::kBFloat16, #x " must be bf16")

namespace hermes {

using bf16 = __nv_bfloat16;

// ----------------------------------------------------------------------------
// device helpers
// ----------------------------------------------------------------------------

__device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16  f2b(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// ----------------------------------------------------------------------------
// kernels
// ----------------------------------------------------------------------------

// LayerNorm: one block per row. blockDim.x must be >= D and a multiple of 32.
template <int D_MAX>
__global__ void layer_norm_kernel(
    const bf16* __restrict__ in,
    const bf16* __restrict__ gamma,
    const bf16* __restrict__ beta,
    bf16* __restrict__ out,
    int N, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;

    const bf16* x = in + row * D;
    bf16* y = out + row * D;

    float v = (tid < D) ? b2f(x[tid]) : 0.0f;

    float s = warp_sum(v);

    __shared__ float ws[D_MAX / 32];
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int n_warps = (D + 31) >> 5;
    if (lane == 0 && warp_id < n_warps) ws[warp_id] = s;
    __syncthreads();

    __shared__ float smean, svar;
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < n_warps; ++i) t += ws[i];
        smean = t / (float)D;
    }
    __syncthreads();

    // Padding threads (tid >= D) must contribute 0 to variance, otherwise
    // diff = -smean and the variance is inflated by smean^2 per pad thread.
    float diff = (tid < D) ? (v - smean) : 0.0f;
    float sq = warp_sum(diff * diff);
    if (lane == 0 && warp_id < n_warps) ws[warp_id] = sq;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < n_warps; ++i) t += ws[i];
        svar = t / (float)D;
    }
    __syncthreads();

    if (tid < D) {
        float g = b2f(gamma[tid]);
        float b = b2f(beta[tid]);
        y[tid] = f2b(diff * rsqrtf(svar + eps) * g + b);
    }
}

// elementwise add bias broadcast over last dim D
__global__ void bias_add_kernel(bf16* __restrict__ x,
                                const bf16* __restrict__ bias,
                                long total, int D)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = b2f(x[i]) + b2f(bias[i % D]);
    x[i] = f2b(v);
}

// fused bias + exact GELU (in-place)
__global__ void bias_gelu_kernel(bf16* __restrict__ x,
                                 const bf16* __restrict__ bias,
                                 long total, int D)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = b2f(x[i]) + b2f(bias[i % D]);
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    x[i] = f2b(v * 0.5f * (1.0f + erff(v * 0.7071067811865476f)));
}

// fused bias + ReLU (in-place)
__global__ void bias_relu_kernel(bf16* __restrict__ x,
                                 const bf16* __restrict__ bias,
                                 long total, int D)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = b2f(x[i]) + b2f(bias[i % D]);
    x[i] = f2b(fmaxf(v, 0.0f));
}

// in-place x += y, total elements
__global__ void residual_add_kernel(bf16* __restrict__ x,
                                    const bf16* __restrict__ y,
                                    long total)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    x[i] = f2b(b2f(x[i]) + b2f(y[i]));
}

// add positional encoding pe[t,0,d] to x[b,t,d]
__global__ void add_pe_kernel(bf16* __restrict__ x,
                              const bf16* __restrict__ pe,
                              int B, int T, int D)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * T * D;
    if (i >= total) return;
    int d = (int)(i % D);
    int t = (int)((i / D) % T);
    float xv = b2f(x[i]);
    float pv = b2f(pe[t * D + d]);
    x[i] = f2b(xv + pv);
}

// embedding lookup: out[b, :] = table[idx[b], :]
// Out-of-range idx -> output zeros (safe fallback, no GPU segfault).
__global__ void emb_gather_kernel(const bf16* __restrict__ table,
                                  const long* __restrict__ idx,
                                  bf16* __restrict__ out,
                                  int B, int E, long V)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B || tid >= E) return;
    long i = idx[b];
    if (i < 0 || i >= V) {
        out[b * E + tid] = f2b(0.0f);
        return;
    }
    out[b * E + tid] = table[i * E + tid];
}

// take last step: out[b, :] = x[b, T-1, :]
__global__ void last_step_kernel(const bf16* __restrict__ x,
                                 bf16* __restrict__ out,
                                 int B, int T, int D)
{
    int b = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || tid >= D) return;
    out[b * D + tid] = x[(b * T + (T - 1)) * D + tid];
}

// gated fusion: main += sigmoid(gate_pre) * aux  (in-place on main)
__global__ void gated_fusion_kernel(bf16* __restrict__ main,
                                    const bf16* __restrict__ aux,
                                    const bf16* __restrict__ gate_pre,
                                    long total)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float m = b2f(main[i]);
    float a = b2f(aux[i]);
    float gp = b2f(gate_pre[i]);
    float g = 1.0f / (1.0f + expf(-gp));
    main[i] = f2b(m + g * a);
}

// ----------------------------------------------------------------------------
// Flash MHA single-block
//
// qkv:  (B, T, 3, H, Dh)  bf16 contiguous
// out:  (B, T, H, Dh)     bf16
//
// Grid: (B, H), Block: T threads (T <= MAX_T)
// Shared: K (T*Dh) + V (T*Dh) in fp32
// Each thread: one query position. Fp32 accumulation. Online-softmax-free
// (single pass since K,V fit in shared mem entirely).
// ----------------------------------------------------------------------------
template <int MAX_T, int DH>
__global__ void flash_mha_kernel(
    const bf16* __restrict__ qkv,
    bf16* __restrict__ out,
    int B, int T, int H, float scale)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int t = threadIdx.x;
    if (t >= T) return;

    extern __shared__ float smem[];
    float* sK = smem;                 // (T, DH)
    float* sV = smem + T * DH;        // (T, DH)

    // pointer base for this (b, h)
    // qkv stride per element: ((b*T + t)*3 + qkv_idx)*H + h, then *Dh + d
    long base = ((long)b * T) * 3 * H * DH + (long)h * DH;
    long stride_t = 3LL * H * DH;

    // Load K[t,h], V[t,h] into shared (cooperative: thread t handles row t)
    long off_K = base + t * stride_t + 1LL * H * DH;
    long off_V = base + t * stride_t + 2LL * H * DH;
#pragma unroll
    for (int d = 0; d < DH; ++d) {
        sK[t * DH + d] = b2f(qkv[off_K + d]);
        sV[t * DH + d] = b2f(qkv[off_V + d]);
    }

    // Load Q[t,h] into register
    float Q[DH];
    long off_Q = base + t * stride_t + 0LL * H * DH;
#pragma unroll
    for (int d = 0; d < DH; ++d) Q[d] = b2f(qkv[off_Q + d]);

    __syncthreads();

    // Compute scores S[j] = Q . K[j] * scale
    float scores[MAX_T];
    float maxv = -INFINITY;
    for (int j = 0; j < T; ++j) {
        float s = 0.0f;
#pragma unroll
        for (int d = 0; d < DH; ++d) s += Q[d] * sK[j * DH + d];
        s *= scale;
        scores[j] = s;
        if (s > maxv) maxv = s;
    }

    // Softmax (numerically stable)
    float sumexp = 0.0f;
    for (int j = 0; j < T; ++j) {
        scores[j] = expf(scores[j] - maxv);
        sumexp += scores[j];
    }
    float inv = 1.0f / sumexp;

    // O = softmax(S) @ V
    float O[DH];
#pragma unroll
    for (int d = 0; d < DH; ++d) O[d] = 0.0f;
    for (int j = 0; j < T; ++j) {
        float w = scores[j] * inv;
#pragma unroll
        for (int d = 0; d < DH; ++d) O[d] += w * sV[j * DH + d];
    }

    // Write out[b, t, h, :]
    long off_O = ((long)b * T + t) * H * DH + (long)h * DH;
#pragma unroll
    for (int d = 0; d < DH; ++d) out[off_O + d] = f2b(O[d]);
}

// ----------------------------------------------------------------------------
// launchers
// ----------------------------------------------------------------------------

static cudaStream_t cur_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

static void launch_layer_norm(torch::Tensor in,
                              torch::Tensor gamma,
                              torch::Tensor beta,
                              torch::Tensor out,
                              float eps = 1e-5f)
{
    int N = in.numel() / in.size(-1);
    int D = (int)in.size(-1);
    TORCH_CHECK(D <= 1024, "LayerNorm: D too large");
    int threads = ((D + 31) / 32) * 32;  // round up to warp
    if (threads < 32) threads = 32;
    auto s = cur_stream();
    layer_norm_kernel<1024><<<N, threads, 0, s>>>(
        (const bf16*)in.data_ptr(),
        (const bf16*)gamma.data_ptr(),
        (const bf16*)beta.data_ptr(),
        (bf16*)out.data_ptr(),
        N, D, eps);
}

static constexpr int EW_THREADS = 1024;  // elementwise block size

static void launch_bias_add(torch::Tensor x, torch::Tensor bias) {
    long total = x.numel();
    int D = (int)bias.size(0);
    int blocks = (int)((total + EW_THREADS - 1) / EW_THREADS);
    bias_add_kernel<<<blocks, EW_THREADS, 0, cur_stream()>>>(
        (bf16*)x.data_ptr(), (const bf16*)bias.data_ptr(), total, D);
}

static void launch_bias_gelu(torch::Tensor x, torch::Tensor bias) {
    long total = x.numel();
    int D = (int)bias.size(0);
    int blocks = (int)((total + EW_THREADS - 1) / EW_THREADS);
    bias_gelu_kernel<<<blocks, EW_THREADS, 0, cur_stream()>>>(
        (bf16*)x.data_ptr(), (const bf16*)bias.data_ptr(), total, D);
}

static void launch_bias_relu(torch::Tensor x, torch::Tensor bias) {
    long total = x.numel();
    int D = (int)bias.size(0);
    int blocks = (int)((total + EW_THREADS - 1) / EW_THREADS);
    bias_relu_kernel<<<blocks, EW_THREADS, 0, cur_stream()>>>(
        (bf16*)x.data_ptr(), (const bf16*)bias.data_ptr(), total, D);
}

static void launch_residual_add(torch::Tensor x, torch::Tensor y) {
    long total = x.numel();
    int blocks = (int)((total + EW_THREADS - 1) / EW_THREADS);
    residual_add_kernel<<<blocks, EW_THREADS, 0, cur_stream()>>>(
        (bf16*)x.data_ptr(), (const bf16*)y.data_ptr(), total);
}

static void launch_add_pe(torch::Tensor x, torch::Tensor pe) {
    int B = (int)x.size(0), T = (int)x.size(1), D = (int)x.size(2);
    long total = (long)B * T * D;
    int blocks = (int)((total + EW_THREADS - 1) / EW_THREADS);
    add_pe_kernel<<<blocks, EW_THREADS, 0, cur_stream()>>>(
        (bf16*)x.data_ptr(), (const bf16*)pe.data_ptr(), B, T, D);
}

static torch::Tensor launch_emb_gather(torch::Tensor table, torch::Tensor idx) {
    TORCH_CHECK(idx.dim() == 1, "emb idx must be 1D, got dim=", idx.dim());
    TORCH_CHECK(table.dim() == 2, "emb table must be 2D");
    int B = (int)idx.size(0);
    int E = (int)table.size(1);
    long V = (long)table.size(0);
    TORCH_CHECK(E <= 1024,
                "emb gather: embed dim ", E, " exceeds 1024 (block thread limit)");
    auto out = torch::zeros({B, E}, table.options());   // zero-init defends against partial writes on bad input
    int threads = ((E + 31) / 32) * 32;
    emb_gather_kernel<<<B, threads, 0, cur_stream()>>>(
        (const bf16*)table.data_ptr(),
        (const long*)idx.data_ptr(),
        (bf16*)out.data_ptr(),
        B, E, V);
    return out;
}

static torch::Tensor launch_last_step(torch::Tensor x) {
    int B = (int)x.size(0), T = (int)x.size(1), D = (int)x.size(2);
    auto out = torch::empty({B, D}, x.options());
    int threads = 256;
    int blocks_y = (D + threads - 1) / threads;
    dim3 grid(B, blocks_y);
    last_step_kernel<<<grid, threads, 0, cur_stream()>>>(
        (const bf16*)x.data_ptr(), (bf16*)out.data_ptr(), B, T, D);
    return out;
}

static void launch_gated_fusion(torch::Tensor main, torch::Tensor aux,
                                torch::Tensor gate_pre)
{
    long total = main.numel();
    int blocks = (int)((total + EW_THREADS - 1) / EW_THREADS);
    gated_fusion_kernel<<<blocks, EW_THREADS, 0, cur_stream()>>>(
        (bf16*)main.data_ptr(),
        (const bf16*)aux.data_ptr(),
        (const bf16*)gate_pre.data_ptr(),
        total);
}

static void launch_flash_mha(torch::Tensor qkv, torch::Tensor out,
                             int B, int T, int H, int Dh)
{
    TORCH_CHECK(Dh == 32, "Hermes engine: Dh must be 32 (D=256, H=8)");
    TORCH_CHECK(T <= 256, "Hermes engine: T must be <= 256");
    float scale = 1.0f / sqrtf((float)Dh);
    dim3 grid(B, H);
    int threads = T;  // one query per thread
    size_t smem_bytes = (size_t)2 * T * Dh * sizeof(float);

    // Blackwell / Hopper need explicit opt-in for >48KB dynamic smem.
    static std::once_flag s_attr_once;
    std::call_once(s_attr_once, [] {
        cudaFuncSetAttribute((void*)flash_mha_kernel<256, 32>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             96 * 1024);
    });

    auto s = cur_stream();
    flash_mha_kernel<256, 32><<<grid, threads, smem_bytes, s>>>(
        (const bf16*)qkv.data_ptr(),
        (bf16*)out.data_ptr(),
        B, T, H, scale);
}

// ----------------------------------------------------------------------------
// Encoder layer
// weight indices (12 per layer, see WEIGHT_KEYS in python):
//   0 norm1.w, 1 norm1.b
//   2 qkv.w,   3 qkv.b
//   4 out.w,   5 out.b
//   6 norm2.w, 7 norm2.b
//   8 lin1.w,  9 lin1.b
//  10 lin2.w, 11 lin2.b
// ----------------------------------------------------------------------------
static torch::Tensor encoder_layer_forward(torch::Tensor x,
                                           const std::vector<torch::Tensor>& w)
{
    int B = (int)x.size(0), T = (int)x.size(1), D = (int)x.size(2);
    int H = 8, Dh = D / H;

    // y = norm1(x)
    auto y = torch::empty_like(x);
    launch_layer_norm(x, w[0], w[1], y);

    // qkv = y @ qkv_w.T + qkv_b   shape (B,T,3D)
    auto qkv = torch::matmul(y, w[2].transpose(0, 1));
    launch_bias_add(qkv, w[3]);
    auto qkv_view = qkv.view({B, T, 3, H, Dh}).contiguous();

    // attention -> (B, T, H, Dh)
    auto attn_o = torch::empty({B, T, H, Dh}, x.options());
    launch_flash_mha(qkv_view, attn_o, B, T, H, Dh);
    auto attn_2d = attn_o.view({B, T, D});

    // attn_proj = attn_2d @ out_w.T + out_b
    auto attn_proj = torch::matmul(attn_2d, w[4].transpose(0, 1));
    launch_bias_add(attn_proj, w[5]);

    // x += attn_proj
    launch_residual_add(x, attn_proj);

    // y2 = norm2(x)
    auto y2 = torch::empty_like(x);
    launch_layer_norm(x, w[6], w[7], y2);

    // ff1 = y2 @ lin1_w.T  ; bias_gelu in-place
    auto ff1 = torch::matmul(y2, w[8].transpose(0, 1));
    launch_bias_gelu(ff1, w[9]);

    // ff2 = ff1 @ lin2_w.T + lin2_b
    auto ff2 = torch::matmul(ff1, w[10].transpose(0, 1));
    launch_bias_add(ff2, w[11]);

    // x += ff2
    launch_residual_add(x, ff2);
    return x;
}

// ----------------------------------------------------------------------------
// Main forward
//
// Weight pack layout (88 tensors), see _build_weight_keys() in inference_engine.py:
//   0: input_projection.weight  (D, 29)
//   1: input_projection.bias    (D,)
//   2: positional_encoding.pe   (max_len, 1, D)  -> we treat as (max_len, D)
//   3: event_projection.0.weight (D, num_events)   [or stub if no event branch]
//   4: event_projection.0.bias   (D,)
//   5: event_gate.gate_proj.weight (D, 2D)
//   6: event_gate.gate_proj.bias   (D,)
//   7..54: 4 layers x 12 weights
//   55: industry_embedding.weight (n_ind, ind_dim=128)
//   56: style_embedding.weight    (n_sty, sty_dim=64)
//   57: regime_embedding.weight   (n_reg, reg_dim=64)
//   58: cg_state_embedding.weight (n_cg,  cg_dim=64)
//   59: risk_regime_embedding.weight (n_rr, rr_dim=64)
//   60: fusion_layer.0.weight (2D, D + cond_dim)
//   61: fusion_layer.0.bias   (2D,)
//   62: fusion_layer.3.weight (D, 2D)
//   63: fusion_layer.3.bias   (D,)
//   64..67: prediction_head 0/3
//   68..71: state_head
//   72..75: score_head
//   76..79: risk_head
//   80..83: signal_head
//   84..87: direction_head
// ----------------------------------------------------------------------------
struct ForwardOutputs {
    torch::Tensor price;
    torch::Tensor cg_state;
    torch::Tensor cg_score;
    torch::Tensor risk_regime;
    torch::Tensor signal;
    torch::Tensor direction;
};

std::vector<torch::Tensor> hermes_forward(
    torch::Tensor x,                              // (B, T, 29) bf16
    torch::Tensor industry_idx,                   // (B,) int64
    torch::Tensor style_idx,
    torch::Tensor regime_idx,
    torch::Tensor cg_state_idx,
    torch::Tensor risk_regime_idx,
    std::optional<torch::Tensor> events,          // (B, T, num_events) bf16
    std::optional<torch::Tensor> texts,           // (B, T, text_dim) bf16  (NOT supported; pass None)
    std::vector<torch::Tensor> weights,
    bool has_event_branch,
    int num_layers)
{
    CHECK_CUDA(x); CHECK_CTG(x); CHECK_BF16(x);
    TORCH_CHECK(x.dim() == 3, "x must be (B, T, in_dim), got dim=", x.dim());
    int B = (int)x.size(0), T = (int)x.size(1);
    TORCH_CHECK(T <= 256, "seq_len T=", T, " exceeds engine limit 256 (single-block flash MHA)");
    // Layout: 7 (preamble) + num_layers*12 (encoder) + 5 (embeddings) +
    //         4 (fusion) + 6*4 (heads) = 7 + 12L + 33  weights total.
    const size_t expected_n = 7 + (size_t)num_layers * 12 + 33;
    TORCH_CHECK(weights.size() == expected_n,
                "weights pack size mismatch: got ", weights.size(),
                ", expected ", expected_n,
                " (num_layers=", num_layers, ")");
    int D = (int)weights[0].size(0);
    int in_dim = (int)weights[0].size(1);
    TORCH_CHECK(x.size(2) == in_dim,
                "x last dim ", x.size(2), " must match input_projection.weight in_dim ", in_dim);
    TORCH_CHECK(D % 8 == 0, "hidden dim D=", D, " must be divisible by 8 heads");
    TORCH_CHECK(D / 8 == 32, "head dim must be 32 (engine fixed for D=256, H=8)");
    TORCH_CHECK(!texts.has_value(), "text branch not supported in this engine version");
    // Note: Hermes attention is full (causal=False) per training time
    // (flash_attention_layer.py:80 uses causal=False; bmm fallback also has no mask).
    // Engine intentionally matches that — do NOT add a causal mask here.

    const c10::cuda::OptionalCUDAGuard guard(device_of(x));

    // 1. input projection: x_proj = x @ in_w.T + in_b   shape (B,T,D)
    auto x_proj = torch::matmul(x, weights[0].transpose(0, 1));
    launch_bias_add(x_proj, weights[1]);

    // 2. event branch (gated fusion)
    if (has_event_branch && events.has_value()) {
        auto ev = events.value();
        CHECK_CUDA(ev); CHECK_CTG(ev); CHECK_BF16(ev);
        // event_proj = GELU(ev @ Wev.T + bev)
        auto ev_proj = torch::matmul(ev, weights[3].transpose(0, 1));
        launch_bias_gelu(ev_proj, weights[4]);
        // gate_pre = cat(x_proj, ev_proj) @ Wg.T + bg
        auto cat_ha = torch::cat({x_proj, ev_proj}, /*dim=*/2);
        auto gate_pre = torch::matmul(cat_ha, weights[5].transpose(0, 1));
        launch_bias_add(gate_pre, weights[6]);
        // x_proj = x_proj + sigmoid(gate_pre) * ev_proj
        launch_gated_fusion(x_proj, ev_proj, gate_pre);
    }

    // 3. positional encoding (pe is (max_len, 1, D), we use its (max_len, D) view)
    auto pe = weights[2].view({-1, D});
    launch_add_pe(x_proj, pe);

    // 4. transformer encoder
    constexpr int W_PER_LAYER = 12;
    int layer_base = 7;
    for (int l = 0; l < num_layers; ++l) {
        std::vector<torch::Tensor> lw;
        lw.reserve(W_PER_LAYER);
        for (int k = 0; k < W_PER_LAYER; ++k) {
            lw.push_back(weights[layer_base + l * W_PER_LAYER + k]);
        }
        x_proj = encoder_layer_forward(x_proj, lw);
    }

    // 5. take last step
    auto last = launch_last_step(x_proj);   // (B, D)

    // 6. conditional embeddings + concat
    int emb_base = layer_base + num_layers * W_PER_LAYER;
    auto ind = launch_emb_gather(weights[emb_base + 0], industry_idx);
    auto sty = launch_emb_gather(weights[emb_base + 1], style_idx);
    auto reg = launch_emb_gather(weights[emb_base + 2], regime_idx);
    auto cgs = launch_emb_gather(weights[emb_base + 3], cg_state_idx);
    auto rsk = launch_emb_gather(weights[emb_base + 4], risk_regime_idx);
    auto cond = torch::cat({ind, sty, reg, cgs, rsk}, /*dim=*/1);
    auto fused_in = torch::cat({last, cond}, /*dim=*/1);

    // 7. fusion_layer: Linear -> ReLU -> (Dropout) -> Linear -> ReLU
    int fus_base = emb_base + 5;
    auto h1 = torch::matmul(fused_in, weights[fus_base + 0].transpose(0, 1));
    launch_bias_relu(h1, weights[fus_base + 1]);
    auto h2 = torch::matmul(h1, weights[fus_base + 2].transpose(0, 1));
    launch_bias_relu(h2, weights[fus_base + 3]);

    // 8. heads — each head: Linear(D, hidden) -> ReLU -> Linear(hidden, out)
    int head_base = fus_base + 4;
    auto run_head = [&](int idx0, /*out_dim*/int) -> torch::Tensor {
        auto wA = weights[idx0 + 0]; auto bA = weights[idx0 + 1];
        auto wB = weights[idx0 + 2]; auto bB = weights[idx0 + 3];
        auto t1 = torch::matmul(h2, wA.transpose(0, 1));
        launch_bias_relu(t1, bA);
        auto t2 = torch::matmul(t1, wB.transpose(0, 1));
        launch_bias_add(t2, bB);
        return t2;
    };

    auto price       = run_head(head_base + 0,  1);
    auto cg_state    = run_head(head_base + 4,  9);
    auto cg_score    = run_head(head_base + 8,  1);
    auto risk_regime = run_head(head_base + 12, 4);
    auto signal      = run_head(head_base + 16, 1);
    auto direction   = run_head(head_base + 20, 2);

    // Fixed order: [price, cg_state, cg_score, risk_regime, signal, direction]
    return {price, cg_state, cg_score, risk_regime, signal, direction};
}

}  // namespace hermes


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hermes_forward", &hermes::hermes_forward,
          "Hermes ConditionalFinancialModel forward (sm_120a, bf16)",
          pybind11::arg("x"),
          pybind11::arg("industry_idx"),
          pybind11::arg("style_idx"),
          pybind11::arg("regime_idx"),
          pybind11::arg("cg_state_idx"),
          pybind11::arg("risk_regime_idx"),
          pybind11::arg("events") = std::nullopt,
          pybind11::arg("texts")  = std::nullopt,
          pybind11::arg("weights"),
          pybind11::arg("has_event_branch") = false,
          pybind11::arg("num_layers") = 4);
}
