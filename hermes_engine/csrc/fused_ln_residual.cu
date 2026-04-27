// fused_ln_residual.cu — out = LayerNorm(x + residual)
//
// Fuses an elementwise add + a row-wise LayerNorm into one kernel.
// Used in every transformer block (LN(x + attn_out), LN(x + ffn_out)).
//
// Saves vs running torch ops separately:
//   - 1 kernel launch instead of 3 (add + reduce mean + normalize)
//   - 2 fewer global-memory round-trips (x+residual stays in registers,
//     normalized output written once)
//
// Layout:
//   in:       (N, D) row-major,    bf16 or fp16
//   residual: (N, D) row-major,    same dtype as in
//   gamma:    (D,)                 same dtype
//   beta:     (D,)                 same dtype
//   out:      (N, D) row-major     same dtype
//
// One block per row. blockDim.x = next pow2 ≥ D, capped at 1024.
// Restrictions:
//   - D ≤ 1024 (one block per row, threads cooperatively load D elements)
//   - D % 32 == 0 strongly preferred for warp-shuffle reductions
//
// For the v1 model D=256 (P1) / 512 (P2) / 768 (P3 — this exceeds 1024-thread
// limit only at very large D; 768 fits in 1024 threads).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace hermes_fast {

#define CHECK_CUDA(x)  TORCH_CHECK(x.is_cuda(),       #x " must be CUDA")
#define CHECK_CTG(x)   TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_2D(x)    TORCH_CHECK(x.dim() == 2,      #x " must be 2D (N, D)")
#define CHECK_1D(x)    TORCH_CHECK(x.dim() == 1,      #x " must be 1D (D,)")

// Warp-level sum reduction (32 lanes).
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, m);
    }
    return v;
}

// Block-level sum reduction. Returns the same value to all threads.
template <int BLOCK_THREADS>
__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    constexpr int N_WARPS = BLOCK_THREADS >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        float w = (threadIdx.x < N_WARPS) ? shared[lane] : 0.0f;
        w = warp_reduce_sum(w);
        if (lane == 0) shared[0] = w;
    }
    __syncthreads();
    return shared[0];
}

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float<__half>          (__half v)           { return __half2float(v);     }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half          from_float<__half>          (float v) { return __float2half(v);     }

template <typename T, int BLOCK_THREADS>
__global__ void fused_ln_residual_kernel(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T*       __restrict__ out,
    int N, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;

    const T* x_row = x        + row * D;
    const T* r_row = residual + row * D;
    T*       o_row = out      + row * D;

    __shared__ float reduce_buf[BLOCK_THREADS / 32];
    __shared__ float mean_shared, rstd_shared;

    // 1) sum and sum-of-squares of (x + residual) — keeps the sum in regs.
    float sum = 0.f, sum_sq = 0.f;
    for (int i = tid; i < D; i += BLOCK_THREADS) {
        float v = to_float(x_row[i]) + to_float(r_row[i]);
        sum    += v;
        sum_sq += v * v;
    }

    sum    = block_reduce_sum<BLOCK_THREADS>(sum,    reduce_buf);
    sum_sq = block_reduce_sum<BLOCK_THREADS>(sum_sq, reduce_buf);

    if (tid == 0) {
        float mean = sum / (float)D;
        float var  = sum_sq / (float)D - mean * mean;
        var = var > 0.f ? var : 0.f;
        mean_shared = mean;
        rstd_shared = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = mean_shared;
    float rstd = rstd_shared;

    // 2) normalize, scale, shift, write.
    for (int i = tid; i < D; i += BLOCK_THREADS) {
        float v   = to_float(x_row[i]) + to_float(r_row[i]);
        float g   = to_float(gamma[i]);
        float b   = to_float(beta[i]);
        float y   = (v - mean) * rstd * g + b;
        o_row[i]  = from_float<T>(y);
    }
}

static int next_pow2_ge(int x, int floor_v = 32, int cap = 1024) {
    int p = floor_v;
    while (p < x && p < cap) p <<= 1;
    if (p < floor_v) p = floor_v;
    if (p > cap)     p = cap;
    return p;
}

torch::Tensor fused_ln_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps)
{
    CHECK_CUDA(x); CHECK_CUDA(residual); CHECK_CUDA(gamma); CHECK_CUDA(beta);
    CHECK_CTG(x);  CHECK_CTG(residual);  CHECK_CTG(gamma);  CHECK_CTG(beta);
    CHECK_2D(x);   CHECK_2D(residual);
    CHECK_1D(gamma); CHECK_1D(beta);

    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have same shape");
    int N = x.size(0);
    int D = x.size(1);
    TORCH_CHECK(gamma.size(0) == D && beta.size(0) == D,
                "gamma/beta must have length D=", D);
    TORCH_CHECK(D <= 1024, "fused_ln_residual: D=", D,
                " exceeds 1024-thread block limit");
    TORCH_CHECK(x.scalar_type() == residual.scalar_type() &&
                x.scalar_type() == gamma.scalar_type() &&
                x.scalar_type() == beta.scalar_type(),
                "x, residual, gamma, beta must share dtype");

    auto out = torch::empty_like(x);
    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int threads = next_pow2_ge(D, 32, 1024);
    dim3 grid(N), block(threads);

    #define DISPATCH(BT)                                                                   \
        if (x.scalar_type() == torch::kBFloat16) {                                          \
            fused_ln_residual_kernel<__nv_bfloat16, BT><<<grid, block, 0, stream>>>(        \
                reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),                            \
                reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),                     \
                reinterpret_cast<__nv_bfloat16*>(gamma.data_ptr()),                        \
                reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),                         \
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),                          \
                N, D, (float)eps);                                                          \
        } else if (x.scalar_type() == torch::kHalf) {                                       \
            fused_ln_residual_kernel<__half, BT><<<grid, block, 0, stream>>>(               \
                reinterpret_cast<__half*>(x.data_ptr()),                                    \
                reinterpret_cast<__half*>(residual.data_ptr()),                             \
                reinterpret_cast<__half*>(gamma.data_ptr()),                                \
                reinterpret_cast<__half*>(beta.data_ptr()),                                 \
                reinterpret_cast<__half*>(out.data_ptr()),                                  \
                N, D, (float)eps);                                                          \
        } else {                                                                            \
            TORCH_CHECK(false, "fused_ln_residual: only bf16/fp16 supported, got ",         \
                        x.scalar_type());                                                   \
        }

    switch (threads) {
        case   32: DISPATCH(  32); break;
        case   64: DISPATCH(  64); break;
        case  128: DISPATCH( 128); break;
        case  256: DISPATCH( 256); break;
        case  512: DISPATCH( 512); break;
        case 1024: DISPATCH(1024); break;
        default: TORCH_CHECK(false, "unsupported block size ", threads);
    }
    #undef DISPATCH

    return out;
}

} // namespace hermes_fast
