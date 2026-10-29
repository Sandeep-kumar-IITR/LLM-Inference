#include "custom_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define WARP_SIZE 32

// Warp reduction helper for float addition
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-wide sum reduction using shared memory
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
        block_sum = shared[lane];
    }
    if (wid == 0) {
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum;
}

// Kernel 1: Fused RMSNorm & Residual Addition
// Out = RMSNorm(Input + Residual) * Gamma
__global__ void fused_rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    float* __restrict__ output_residual,
    int hidden_dim,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_input = input + row * hidden_dim;
    const float* row_res = (residual != nullptr) ? (residual + row * hidden_dim) : nullptr;
    float* row_out = output + row * hidden_dim;
    float* row_out_res = (output_residual != nullptr) ? (output_residual + row * hidden_dim) : nullptr;

    // 1. Accumulate sum of squares across thread elements
    float sum_sq = 0.0f;
    for (int col = tid; col < hidden_dim; col += blockDim.x) {
        float val = row_input[col];
        if (row_res != nullptr) {
            val += row_res[col];
        }
        if (row_out_res != nullptr) {
            row_out_res[col] = val; // Store updated residual
        }
        sum_sq += val * val;
    }

    // 2. Reduce sum of squares across entire block
    __shared__ float s_variance_inv;
    float block_sum_sq = block_reduce_sum(sum_sq);

    if (tid == 0) {
        float mean_sq = block_sum_sq / (float)hidden_dim;
        s_variance_inv = rsqrtf(mean_sq + epsilon);
    }
    __syncthreads();

    float inv_rms = s_variance_inv;

    // 3. Normalize and scale by gamma
    for (int col = tid; col < hidden_dim; col += blockDim.x) {
        float val = (row_out_res != nullptr) ? row_out_res[col] : (row_input[col] + (row_res ? row_res[col] : 0.0f));
        float g = (gamma != nullptr) ? gamma[col] : 1.0f;
        row_out[col] = val * inv_rms * g;
    }
}

void launch_fused_rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    float* __restrict__ output_residual,
    int batch_size,
    int hidden_dim,
    float epsilon,
    cudaStream_t stream
) {
    int threads = (hidden_dim < 1024) ? min(hidden_dim, 256) : 512;
    // Align threads to warp size
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    fused_rmsnorm_kernel<<<batch_size, threads, 0, stream>>>(
        input, residual, gamma, output, output_residual, hidden_dim, epsilon
    );
}

// Kernel 2: Fused SiLU and Mul (SwiGLU)
// Given input of shape [..., 2 * hidden_dim], split into gate and up:
// output[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
__global__ void silu_and_mul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_rows,
    int hidden_dim
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_elements = total_rows * hidden_dim;

    if (idx < total_elements) {
        int row = idx / hidden_dim;
        int col = idx % hidden_dim;

        int gate_idx = row * (2 * hidden_dim) + col;
        int up_idx = gate_idx + hidden_dim;

        float gate = input[gate_idx];
        float up = input[up_idx];

        // SiLU: gate / (1.0f + expf(-gate))
        float silu = gate / (1.0f + expf(-gate));
        output[idx] = silu * up;
    }
}

void launch_silu_and_mul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_rows,
    int hidden_dim,
    cudaStream_t stream
) {
    int total_elements = total_rows * hidden_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    silu_and_mul_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_rows, hidden_dim
    );
}
