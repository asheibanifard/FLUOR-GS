/*
 * splat_mip_tiled_cuda.cu — Tile-based CUDA kernel for MIP splatting
 *
 * Tile-based rasterization for MIP (Maximum Intensity Projection) splatting.
 * Instead of each pixel thread scanning ALL K Gaussians, this approach:
 *   1. Precomputes which Gaussians overlap each 16×16 tile (Python preprocessing)
 *   2. Sorts Gaussian indices by tile and builds per-tile offset table
 *   3. Launches one thread block per tile; threads cooperatively load
 *      Gaussian data into shared memory in batches, then each thread
 *      evaluates its pixel against only the loaded batch
 *
 * Complexity: O(K_tile) per pixel instead of O(K), where K_tile << K.
 * With K=50,000 total Gaussians, typical K_tile ≈ 50-200 per tile.
 *
 * MIP formula per pixel:
 *     g_k   = intensity_k * exp(-0.5 * mahalanobis_k(pixel))
 *     out   = Σ_k  softmax(β·g)_k · g_k
 *
 * Backward: ∂out/∂g_k = sm_k · [1 + β(g_k − out)]
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define TILE_SIZE 16
#define BLOCK_THREADS (TILE_SIZE * TILE_SIZE)   // 256
#define SHARED_BATCH  256   // == BLOCK_THREADS; each thread loads one Gaussian
#define MAHAL_CUTOFF  16.0f // skip exp(-8) ≈ 3.4e-4


// ============================================================================
//  Tiled Forward Kernel — one block per tile, shared-memory batching
// ============================================================================
__global__ void tiled_mip_forward_kernel(
    const float* __restrict__ means_2d,           // (K, 2)
    const float* __restrict__ cov_inv,            // (K, 3) packed [a, b, d]
    const float* __restrict__ intensities,        // (K,)
    const int*   __restrict__ sorted_gauss_ids,   // (P,) Gaussian indices sorted by tile
    const int*   __restrict__ tile_offsets,        // (T+1,)  tile t owns [tile_offsets[t], tile_offsets[t+1])
    float*       __restrict__ rendered,            // (H*W,)
    float*       __restrict__ out_max_bg,          // (H*W,)  saved for backward
    float*       __restrict__ out_sum_exp,         // (H*W,)  saved for backward
    const int H,
    const int W,
    const int n_tiles_x,
    const float beta
) {
    // ── tile & pixel coordinates ───────────────────────────────────
    const int tile_idx = blockIdx.x;
    const int tile_y   = tile_idx / n_tiles_x;
    const int tile_x   = tile_idx % n_tiles_x;

    const int local_x = threadIdx.x % TILE_SIZE;
    const int local_y = threadIdx.x / TILE_SIZE;

    const int px_x = tile_x * TILE_SIZE + local_x;
    const int px_y = tile_y * TILE_SIZE + local_y;
    const bool inside = (px_x < W && px_y < H);

    const float px = (float)px_x + 0.5f;
    const float py = (float)px_y + 0.5f;

    // ── online softmax state ───────────────────────────────────────
    float m = -1e30f;   // running max of β·g_k
    float S = 0.0f;     // Σ exp(β·g_k − m)
    float W_acc = 0.0f; // Σ exp(β·g_k − m) · g_k

    // ── tile's Gaussian range ──────────────────────────────────────
    const int range_start = tile_offsets[tile_idx];
    const int range_end   = tile_offsets[tile_idx + 1];
    const int range_count = range_end - range_start;

    // ── shared memory for batch loading ────────────────────────────
    __shared__ float s_mean_x[SHARED_BATCH];
    __shared__ float s_mean_y[SHARED_BATCH];
    __shared__ float s_inv_a[SHARED_BATCH];
    __shared__ float s_inv_b[SHARED_BATCH];
    __shared__ float s_inv_d[SHARED_BATCH];
    __shared__ float s_intensity[SHARED_BATCH];

    // ── process Gaussians in batches ───────────────────────────────
    for (int batch_off = 0; batch_off < range_count; batch_off += SHARED_BATCH) {
        const int batch_size = min(SHARED_BATCH, range_count - batch_off);

        // Each of 256 threads loads one Gaussian (SHARED_BATCH == BLOCK_THREADS)
        if (threadIdx.x < batch_size) {
            const int gid = sorted_gauss_ids[range_start + batch_off + threadIdx.x];
            s_mean_x[threadIdx.x]    = means_2d[gid * 2 + 0];
            s_mean_y[threadIdx.x]    = means_2d[gid * 2 + 1];
            s_inv_a[threadIdx.x]     = cov_inv[gid * 3 + 0];
            s_inv_b[threadIdx.x]     = cov_inv[gid * 3 + 1];
            s_inv_d[threadIdx.x]     = cov_inv[gid * 3 + 2];
            s_intensity[threadIdx.x] = intensities[gid];
        }
        __syncthreads();

        // Each thread evaluates the loaded Gaussians for its pixel
        if (inside) {
            for (int j = 0; j < batch_size; ++j) {
                const float dx = px - s_mean_x[j];
                const float dy = py - s_mean_y[j];

                const float mahal = s_inv_a[j] * dx * dx
                                  + 2.0f * s_inv_b[j] * dx * dy
                                  + s_inv_d[j] * dy * dy;
                if (mahal > MAHAL_CUTOFF) continue;

                const float g  = s_intensity[j] * expf(-0.5f * mahal);
                const float bg = beta * g;

                // Online softmax accumulation
                if (bg > m) {
                    const float corr = expf(m - bg);
                    S     = S * corr + 1.0f;
                    W_acc = W_acc * corr + g;
                    m     = bg;
                } else {
                    const float e = expf(bg - m);
                    S     += e;
                    W_acc += e * g;
                }
            }
        }
        __syncthreads();   // barrier before next batch overwrites shared mem
    }

    // ── write output ───────────────────────────────────────────────
    if (inside) {
        const int pix_idx = px_y * W + px_x;
        const float out_val = (S > 0.0f) ? W_acc / S : 0.0f;
        rendered[pix_idx]    = out_val;
        out_max_bg[pix_idx]  = m;
        out_sum_exp[pix_idx] = S;
    }
}


// ============================================================================
//  Tiled Backward Kernel — one block per tile, atomicAdd to per-Gaussian grads
// ============================================================================
//
//  out  = Σ_k sm_k · g_k     where  sm_k = exp(β g_k) / Σ_j exp(β g_j)
//
//  ∂out/∂g_k = sm_k · [1 + β (g_k − out)]
//
//  g_k = intensity_k · exp(−½ mahal_k)
//    ∂g_k/∂intensity_k = exp(−½ mahal_k)
//    ∂g_k/∂mahal_k     = g_k · (−½)
//
//  mahal = ia·dx² + 2·ib·dx·dy + id·dy²
//    ∂mahal/∂(ia)     =  dx²
//    ∂mahal/∂(ib)     =  2 dx dy
//    ∂mahal/∂(id)     =  dy²
//    ∂mahal/∂(mean_x) = −(2 ia dx + 2 ib dy)
//    ∂mahal/∂(mean_y) = −(2 ib dx + 2 id dy)
//
__global__ void tiled_mip_backward_kernel(
    const float* __restrict__ dL_drendered,       // (H*W,)
    const float* __restrict__ means_2d,           // (K, 2)
    const float* __restrict__ cov_inv,            // (K, 3)
    const float* __restrict__ intensities,        // (K,)
    const int*   __restrict__ sorted_gauss_ids,   // (P,)
    const int*   __restrict__ tile_offsets,        // (T+1,)
    const float* __restrict__ rendered,            // (H*W,)  forward output
    const float* __restrict__ fwd_max_bg,          // (H*W,)
    const float* __restrict__ fwd_sum_exp,         // (H*W,)
    float*       __restrict__ grad_means_2d,       // (K, 2)
    float*       __restrict__ grad_cov_inv,        // (K, 3)
    float*       __restrict__ grad_intensities,    // (K,)
    const int H,
    const int W,
    const int n_tiles_x,
    const float beta
) {
    // ── tile & pixel coordinates ───────────────────────────────────
    const int tile_idx = blockIdx.x;
    const int tile_y   = tile_idx / n_tiles_x;
    const int tile_x   = tile_idx % n_tiles_x;

    const int local_x = threadIdx.x % TILE_SIZE;
    const int local_y = threadIdx.x / TILE_SIZE;

    const int px_x = tile_x * TILE_SIZE + local_x;
    const int px_y = tile_y * TILE_SIZE + local_y;
    const bool inside = (px_x < W && px_y < H);

    const float px = (float)px_x + 0.5f;
    const float py = (float)px_y + 0.5f;

    // ── load per-pixel forward data ────────────────────────────────
    float dL_dout = 0.0f, out_val = 0.0f, lse = 0.0f;
    bool has_work = false;
    if (inside) {
        const int pix_idx = px_y * W + px_x;
        dL_dout = dL_drendered[pix_idx];
        out_val = rendered[pix_idx];
        const float m_val = fwd_max_bg[pix_idx];
        const float s_val = fwd_sum_exp[pix_idx];
        has_work = (fabsf(dL_dout) > 1e-12f) && (s_val > 0.0f);
        if (has_work) {
            lse = m_val + logf(fmaxf(s_val, 1e-30f));
        }
    }

    // ── tile's Gaussian range ──────────────────────────────────────
    const int range_start = tile_offsets[tile_idx];
    const int range_end   = tile_offsets[tile_idx + 1];
    const int range_count = range_end - range_start;

    // ── shared memory ──────────────────────────────────────────────
    __shared__ float s_mean_x[SHARED_BATCH];
    __shared__ float s_mean_y[SHARED_BATCH];
    __shared__ float s_inv_a[SHARED_BATCH];
    __shared__ float s_inv_b[SHARED_BATCH];
    __shared__ float s_inv_d[SHARED_BATCH];
    __shared__ float s_intensity[SHARED_BATCH];
    __shared__ int   s_gid[SHARED_BATCH];

    // ── process Gaussians in batches ───────────────────────────────
    for (int batch_off = 0; batch_off < range_count; batch_off += SHARED_BATCH) {
        const int batch_size = min(SHARED_BATCH, range_count - batch_off);

        // Cooperatively load batch
        if (threadIdx.x < batch_size) {
            const int gid = sorted_gauss_ids[range_start + batch_off + threadIdx.x];
            s_gid[threadIdx.x]       = gid;
            s_mean_x[threadIdx.x]    = means_2d[gid * 2 + 0];
            s_mean_y[threadIdx.x]    = means_2d[gid * 2 + 1];
            s_inv_a[threadIdx.x]     = cov_inv[gid * 3 + 0];
            s_inv_b[threadIdx.x]     = cov_inv[gid * 3 + 1];
            s_inv_d[threadIdx.x]     = cov_inv[gid * 3 + 2];
            s_intensity[threadIdx.x] = intensities[gid];
        }
        __syncthreads();

        if (has_work) {
            for (int j = 0; j < batch_size; ++j) {
                const float dx = px - s_mean_x[j];
                const float dy = py - s_mean_y[j];

                const float ia = s_inv_a[j];
                const float ib = s_inv_b[j];
                const float id = s_inv_d[j];

                const float mahal = ia * dx * dx
                                  + 2.0f * ib * dx * dy
                                  + id * dy * dy;
                if (mahal > MAHAL_CUTOFF) continue;

                const float gauss_exp = expf(-0.5f * mahal);
                const float g  = s_intensity[j] * gauss_exp;
                const float bg = beta * g;

                // softmax weight: sm_k = exp(β g_k − lse)
                const float sm = expf(bg - lse);

                // ∂out/∂g_k = sm_k · [1 + β(g_k − out)]
                const float dout_dg = sm * (1.0f + beta * (g - out_val));
                const float dL_dg   = dL_dout * dout_dg;

                const int gid = s_gid[j];

                // ∂g/∂intensity = exp(−½ mahal)
                atomicAdd(&grad_intensities[gid], dL_dg * gauss_exp);

                // ∂g/∂mahal = g · (−½)
                const float dL_dmahal = dL_dg * g * (-0.5f);

                // ∂mahal/∂(cov_inv)
                atomicAdd(&grad_cov_inv[gid * 3 + 0], dL_dmahal * dx * dx);
                atomicAdd(&grad_cov_inv[gid * 3 + 1], dL_dmahal * 2.0f * dx * dy);
                atomicAdd(&grad_cov_inv[gid * 3 + 2], dL_dmahal * dy * dy);

                // ∂mahal/∂(mean)  — note negative from dx = px − mean_x
                const float dmahal_ddx = 2.0f * ia * dx + 2.0f * ib * dy;
                const float dmahal_ddy = 2.0f * ib * dx + 2.0f * id * dy;
                atomicAdd(&grad_means_2d[gid * 2 + 0], -dL_dmahal * dmahal_ddx);
                atomicAdd(&grad_means_2d[gid * 2 + 1], -dL_dmahal * dmahal_ddy);
            }
        }
        __syncthreads();
    }
}


// ============================================================================
//  C++ Interface
// ============================================================================

std::vector<torch::Tensor> tiled_mip_forward(
    torch::Tensor means_2d,           // (K, 2)
    torch::Tensor cov_inv,            // (K, 3)
    torch::Tensor intensities,        // (K,)
    torch::Tensor sorted_gauss_ids,   // (P,) int32
    torch::Tensor tile_offsets,        // (T+1,) int32
    int H, int W,
    int n_tiles_x, int n_tiles_y,
    float beta
) {
    means_2d         = means_2d.contiguous();
    cov_inv          = cov_inv.contiguous();
    intensities      = intensities.contiguous();
    sorted_gauss_ids = sorted_gauss_ids.contiguous();
    tile_offsets     = tile_offsets.contiguous();

    auto opts     = means_2d.options();   // float32, same device
    auto rendered = torch::zeros({H * W}, opts);
    auto max_bg   = torch::full({H * W}, -1e30f, opts);
    auto sum_exp  = torch::zeros({H * W}, opts);

    const int n_tiles = n_tiles_x * n_tiles_y;

    if (n_tiles > 0 && means_2d.size(0) > 0) {
        tiled_mip_forward_kernel<<<n_tiles, BLOCK_THREADS>>>(
            means_2d.data_ptr<float>(),
            cov_inv.data_ptr<float>(),
            intensities.data_ptr<float>(),
            sorted_gauss_ids.data_ptr<int>(),
            tile_offsets.data_ptr<int>(),
            rendered.data_ptr<float>(),
            max_bg.data_ptr<float>(),
            sum_exp.data_ptr<float>(),
            H, W, n_tiles_x, beta
        );
    }

    return {rendered, max_bg, sum_exp};
}


std::vector<torch::Tensor> tiled_mip_backward(
    torch::Tensor dL_drendered,        // (H*W,)
    torch::Tensor means_2d,            // (K, 2)
    torch::Tensor cov_inv,             // (K, 3)
    torch::Tensor intensities,         // (K,)
    torch::Tensor sorted_gauss_ids,    // (P,) int32
    torch::Tensor tile_offsets,         // (T+1,) int32
    torch::Tensor rendered,             // (H*W,)
    torch::Tensor fwd_max_bg,           // (H*W,)
    torch::Tensor fwd_sum_exp,          // (H*W,)
    int H, int W,
    int n_tiles_x, int n_tiles_y,
    float beta
) {
    dL_drendered     = dL_drendered.contiguous();
    means_2d         = means_2d.contiguous();
    cov_inv          = cov_inv.contiguous();
    intensities      = intensities.contiguous();
    sorted_gauss_ids = sorted_gauss_ids.contiguous();
    tile_offsets     = tile_offsets.contiguous();
    rendered         = rendered.contiguous();
    fwd_max_bg       = fwd_max_bg.contiguous();
    fwd_sum_exp      = fwd_sum_exp.contiguous();

    const int K = means_2d.size(0);

    auto grad_means_2d    = torch::zeros({K, 2}, means_2d.options());
    auto grad_cov_inv     = torch::zeros({K, 3}, cov_inv.options());
    auto grad_intensities = torch::zeros({K},    intensities.options());

    const int n_tiles = n_tiles_x * n_tiles_y;

    if (n_tiles > 0 && K > 0) {
        tiled_mip_backward_kernel<<<n_tiles, BLOCK_THREADS>>>(
            dL_drendered.data_ptr<float>(),
            means_2d.data_ptr<float>(),
            cov_inv.data_ptr<float>(),
            intensities.data_ptr<float>(),
            sorted_gauss_ids.data_ptr<int>(),
            tile_offsets.data_ptr<int>(),
            rendered.data_ptr<float>(),
            fwd_max_bg.data_ptr<float>(),
            fwd_sum_exp.data_ptr<float>(),
            grad_means_2d.data_ptr<float>(),
            grad_cov_inv.data_ptr<float>(),
            grad_intensities.data_ptr<float>(),
            H, W, n_tiles_x, beta
        );
    }

    return {grad_means_2d, grad_cov_inv, grad_intensities};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &tiled_mip_forward,  "Tiled MIP splatting forward  (CUDA)");
    m.def("backward", &tiled_mip_backward, "Tiled MIP splatting backward (CUDA)");
}
