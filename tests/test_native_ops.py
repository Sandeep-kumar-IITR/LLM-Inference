"""Tests for native C++ and CUDA operators against PyTorch reference implementations."""

import pytest
import torch


# Reference Implementations

def ref_rmsnorm(x: torch.Tensor, residual: torch.Tensor = None, gamma: torch.Tensor = None, eps: float = 1e-6):
    """PyTorch reference RMSNorm with optional residual addition."""
    if residual is not None:
        x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    if gamma is not None:
        normed = normed * gamma
    return normed


def ref_silu_and_mul(x: torch.Tensor):
    """PyTorch reference SwiGLU: SiLU(gate) * up."""
    hidden_dim = x.shape[-1] // 2
    gate = x[..., :hidden_dim]
    up = x[..., hidden_dim:]
    return torch.nn.functional.silu(gate) * up


def ref_prioritize(request_ids, arrival_times, token_counts, policy="fcfs"):
    """Python reference priority sorting."""
    zipped = list(zip(request_ids, arrival_times, token_counts))
    if policy == "sjf":
        zipped.sort(key=lambda item: (item[2], item[1]))
    else:
        zipped.sort(key=lambda item: item[1])
    return [item[0] for item in zipped]


class TestNativeOps:
    """Test suite for native C++ and CUDA operations."""

    def test_python_reference_rmsnorm(self):
        """Sanity check reference RMSNorm logic."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 128)
        gamma = torch.ones(128)
        out = ref_rmsnorm(x, gamma=gamma)
        assert out.shape == x.shape
        # Mean square of output should be approximately 1
        mean_sq = out.pow(2).mean(-1)
        assert torch.allclose(mean_sq, torch.ones_like(mean_sq), atol=1e-2)

    def test_python_reference_silu_and_mul(self):
        """Sanity check reference SwiGLU logic."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 256)
        out = ref_silu_and_mul(x)
        assert out.shape == (2, 4, 128)

    def test_python_reference_prioritization(self):
        ids = ["req1", "req2", "req3"]
        arrivals = [10.5, 9.0, 12.0]
        tokens = [50, 100, 20]

        # FCFS should sort by arrival: req2 (9.0), req1 (10.5), req3 (12.0)
        sorted_fcfs = ref_prioritize(ids, arrivals, tokens, policy="fcfs")
        assert sorted_fcfs == ["req2", "req1", "req3"]

        # SJF should sort by tokens: req3 (20), req1 (50), req2 (100)
        sorted_sjf = ref_prioritize(ids, arrivals, tokens, policy="sjf")
        assert sorted_sjf == ["req3", "req1", "req2"]

    @pytest.mark.native
    def test_native_c_plus_plus_prioritization(self):
        """Test compiled C++ prioritization if available."""
        try:
            import llm_engine_native
        except ImportError:
            pytest.skip("llm_engine_native not compiled yet")

        ids = ["reqA", "reqB", "reqC"]
        arrivals = [5.0, 2.0, 8.0]
        tokens = [100, 300, 50]

        res = llm_engine_native.prioritize_requests(ids, arrivals, tokens, "fcfs")
        expected = ref_prioritize(ids, arrivals, tokens, "fcfs")
        assert res == expected

    @pytest.mark.gpu
    @pytest.mark.native
    def test_cuda_fused_rmsnorm(self):
        """Test CUDA kernel vs PyTorch reference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            import llm_engine_native
        except ImportError:
            pytest.skip("llm_engine_native not compiled yet")

        torch.manual_seed(42)
        batch, seq, hidden_dim = 4, 16, 896  # Qwen2.5-0.5B hidden_size = 896
        x = torch.randn(batch, seq, hidden_dim, device="cuda", dtype=torch.float32)
        residual = torch.randn_like(x)
        gamma = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)

        # PyTorch reference
        ref_out = ref_rmsnorm(x, residual=residual, gamma=gamma, eps=1e-6)

        # Native CUDA kernel
        native_outs = llm_engine_native.fused_rmsnorm(x, residual, gamma, 1e-6)
        native_out = native_outs[0]

        assert torch.allclose(native_out, ref_out, atol=1e-3, rtol=1e-3), (
            f"Max diff: {(native_out - ref_out).abs().max().item()}"
        )

    @pytest.mark.gpu
    @pytest.mark.native
    def test_cuda_silu_and_mul(self):
        """Test CUDA SwiGLU kernel vs PyTorch reference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            import llm_engine_native
        except ImportError:
            pytest.skip("llm_engine_native not compiled yet")

        torch.manual_seed(42)
        batch, seq, intermediate_dim = 2, 8, 4864 * 2  # Qwen intermediate dim * 2
        x = torch.randn(batch, seq, intermediate_dim, device="cuda", dtype=torch.float32)

        ref_out = ref_silu_and_mul(x)
        native_out = llm_engine_native.silu_and_mul(x)

        assert torch.allclose(native_out, ref_out, atol=1e-3, rtol=1e-3), (
            f"Max diff: {(native_out - ref_out).abs().max().item()}"
        )
