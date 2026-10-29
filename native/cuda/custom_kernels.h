#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA Kernel Declarations
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
);

void launch_silu_and_mul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_elements,
    int hidden_dim,
    cudaStream_t stream
);
