#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "custom_kernels.h"

// PyTorch Op: Fused RMSNorm & Residual Addition
std::vector<torch::Tensor> fused_rmsnorm_cuda(
    torch::Tensor input,
    c10::optional<torch::Tensor> residual,
    c10::optional<torch::Tensor> gamma,
    float epsilon
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");

    int hidden_dim = input.size(-1);
    int batch_size = input.numel() / hidden_dim;

    const float* residual_ptr = nullptr;
    torch::Tensor output_residual;
    if (residual.has_value() && residual.value().defined()) {
        auto res = residual.value();
        TORCH_CHECK(res.is_cuda(), "residual must be a CUDA tensor");
        TORCH_CHECK(res.is_contiguous(), "residual must be contiguous");
        TORCH_CHECK(res.sizes() == input.sizes(), "residual must match input shape");
        residual_ptr = res.data_ptr<float>();
        output_residual = torch::empty_like(res);
    }

    const float* gamma_ptr = nullptr;
    if (gamma.has_value() && gamma.value().defined()) {
        auto g = gamma.value();
        TORCH_CHECK(g.is_cuda(), "gamma must be a CUDA tensor");
        TORCH_CHECK(g.is_contiguous(), "gamma must be contiguous");
        TORCH_CHECK(g.numel() == hidden_dim, "gamma must have size equal to hidden_dim");
        gamma_ptr = g.data_ptr<float>();
    }

    auto output = torch::empty_like(input);
    float* out_res_ptr = output_residual.defined() ? output_residual.data_ptr<float>() : nullptr;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_fused_rmsnorm_kernel(
        input.data_ptr<float>(),
        residual_ptr,
        gamma_ptr,
        output.data_ptr<float>(),
        out_res_ptr,
        batch_size,
        hidden_dim,
        epsilon,
        stream
    );

    AT_CUDA_CHECK(cudaGetLastError());

    if (output_residual.defined()) {
        return {output, output_residual};
    } else {
        return {output};
    }
}

// PyTorch Op: Fused SiLU and Mul
torch::Tensor silu_and_mul_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");

    int last_dim = input.size(-1);
    TORCH_CHECK(last_dim % 2 == 0, "Last dimension must be divisible by 2");
    int hidden_dim = last_dim / 2;
    int total_rows = input.numel() / last_dim;

    auto out_sizes = input.sizes().vec();
    out_sizes.back() = hidden_dim;
    auto output = torch::empty(out_sizes, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_silu_and_mul_kernel(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_rows,
        hidden_dim,
        stream
    );

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}
