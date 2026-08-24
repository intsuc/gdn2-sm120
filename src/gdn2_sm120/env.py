"""Print the runtime versions that affect CuTe DSL code generation."""

from __future__ import annotations


def main() -> None:
    import cutlass
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"torch={torch.__version__}")
    print(f"cutlass={getattr(cutlass, '__version__', 'unknown')}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={props.name}")
    print(f"compute_capability={props.major}.{props.minor}")
    print(f"sm_count={props.multi_processor_count}")
    print(f"memory_gib={props.total_memory / 2**30:.1f}")


if __name__ == "__main__":
    main()
