import runpy
import sys
import types
from pathlib import Path

import torch


DEEPSEEK_INFERENCE_DIR = Path("/zhutingqi/deepseek v4/inference")


def main() -> None:
    # The FP8-converted checkpoint does not use FP4 expert weights, but the
    # reference model code still compares against this dtype unconditionally.
    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = object()  # type: ignore[attr-defined]
    if "fast_hadamard_transform" not in sys.modules:
        module = types.ModuleType("fast_hadamard_transform")

        def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            y = x
            n = y.size(-1)
            if n & (n - 1) != 0:
                raise ValueError(f"Hadamard fallback requires power-of-two last dim, got {n}")
            h = 1
            orig_shape = y.shape
            y = y.reshape(-1, n)
            while h < n:
                y = y.reshape(-1, n // (h * 2), h * 2)
                left = y[..., :h]
                right = y[..., h : h * 2]
                y = torch.cat((left + right, left - right), dim=-1)
                h *= 2
            return y.reshape(orig_shape) * scale

        module.hadamard_transform = hadamard_transform
        sys.modules["fast_hadamard_transform"] = module
    sys.path.insert(0, str(DEEPSEEK_INFERENCE_DIR))
    import model as deepseek_model

    def compatible_linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
        assert bias is None
        if weight.dtype == torch.float4_e2m1fn_x2:
            x, s = deepseek_model.act_quant(
                x, deepseek_model.block_size, deepseek_model.scale_fmt, deepseek_model.scale_dtype
            )
            weight_scale = weight.scale.float() if deepseek_model.scale_dtype == torch.float32 else weight.scale
            return deepseek_model.fp4_gemm(x, s, weight, weight_scale, deepseek_model.scale_dtype)
        if weight.dtype == torch.float8_e4m3fn:
            x, s = deepseek_model.act_quant(
                x, deepseek_model.block_size, deepseek_model.scale_fmt, deepseek_model.scale_dtype
            )
            weight_scale = weight.scale.float() if deepseek_model.scale_dtype == torch.float32 else weight.scale
            return deepseek_model.fp8_gemm(x, s, weight, weight_scale, deepseek_model.scale_dtype)
        return torch.nn.functional.linear(x, weight)

    deepseek_model.linear = compatible_linear
    runpy.run_path(str(DEEPSEEK_INFERENCE_DIR / "generate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
