#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Forward declarations
std::vector<torch::Tensor> fused_rmsnorm_cuda(
    torch::Tensor input,
    c10::optional<torch::Tensor> residual,
    c10::optional<torch::Tensor> gamma,
    float epsilon
);

torch::Tensor silu_and_mul_cuda(torch::Tensor input);

std::vector<std::string> prioritize_requests_cpp(
    const std::vector<std::string>& request_ids,
    const std::vector<double>& arrival_times,
    const std::vector<int>& token_counts,
    const std::string& policy
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Native C++ and CUDA extensions for High-Performance LLM Inference";

    // CUDA ops
    m.def(
        "fused_rmsnorm",
        &fused_rmsnorm_cuda,
        "Fused RMSNorm and Residual Addition (CUDA)",
        py::arg("input"),
        py::arg("residual") = py::none(),
        py::arg("gamma") = py::none(),
        py::arg("epsilon") = 1e-6
    );

    m.def(
        "silu_and_mul",
        &silu_and_mul_cuda,
        "Fused SiLU and Mul (SwiGLU) activation (CUDA)",
        py::arg("input")
    );

    // C++ ops with GIL release
    m.def(
        "prioritize_requests",
        &prioritize_requests_cpp,
        "Prioritize requests using C++ sorting with GIL released",
        py::arg("request_ids"),
        py::arg("arrival_times"),
        py::arg("token_counts"),
        py::arg("policy") = "fcfs"
    );
}
