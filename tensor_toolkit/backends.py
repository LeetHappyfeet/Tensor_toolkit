"""Execution backend policy.

Version 0.2 supports only the NumPy/CPU reference implementation. GPU execution is
intentionally sidelined until a real, independently validated backend exists.
"""
SUPPORTED_BACKENDS = ("cpu",)


def require_backend(name: str) -> str:
    name = name.lower()
    if name == "gpu":
        raise NotImplementedError(
            "GPU execution is not supported in Tensor Toolkit 0.2; use --backend cpu. "
            "GPU support is reserved for a future validated backend."
        )
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported backend {name!r}; supported: {', '.join(SUPPORTED_BACKENDS)}"
        )
    return name
