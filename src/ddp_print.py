import builtins
import os

try:
    import torch.distributed as dist
except ImportError:
    dist = None

_ORIGINAL_PRINT = builtins.print


def _get_rank() -> int:
    """
    Try very hard to get a global rank that works with:
    - torchrun / deepspeed (RANK)
    - accelerate (ACCELERATE_PROCESS_INDEX)
    - plain torch.distributed
    """
    # 1) torchrun / deepspeed
    if "RANK" in os.environ:
        try:
            return int(os.environ["RANK"])
        except ValueError:
            pass

    # 2) accelerate
    if "ACCELERATE_PROCESS_INDEX" in os.environ:
        try:
            return int(os.environ["ACCELERATE_PROCESS_INDEX"])
        except ValueError:
            pass

    # 3) fallback to torch.distributed
    if dist is not None and dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank()
        except Exception:
            pass

    # Single process / unknown: treat as rank 0
    return 0


def _print_rank0(*args, **kwargs):
    """
    Drop-in replacement for print.

    Extra kwarg:
        force=True  -> print from all ranks
    """
    force = kwargs.pop("force", False)
    if force or _get_rank() == 0:
        _ORIGINAL_PRINT(*args, **kwargs)


def patch_print():
    """Patch builtins.print so only rank 0 prints by default."""
    builtins.print = _print_rank0


def unpatch_print():
    """Restore the original builtins.print."""
    builtins.print = _ORIGINAL_PRINT
