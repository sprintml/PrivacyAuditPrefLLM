import glob
import os
import pickle
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from joblib import Parallel, delayed


def _load_one_shadow(shadow_id: int, path: str) -> Tuple[int, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return shadow_id, obj


def _rss_gib() -> Optional[float]:
    status_path = "/proc/self/status"
    try:
        with open(status_path, "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def _format_gib(num_bytes: int) -> str:
    return f"{float(num_bytes) / (1024.0 ** 3):.2f} GiB"


def load_shadow_pickles(
    dir_path: str,
    n_jobs: int = -1,
    backend: str = "threading",
    check_contiguous: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
    log_prefix: str = "[shadow-load]",
    batch_size: Optional[int] = None,
    progress_every: int = 8,
    shadow_ids: Optional[List[int]] = None,
    max_shadows: Optional[int] = None,
) -> Dict[int, Any]:
    """
    Load all shadow_*.pkl files from a directory into a dict keyed by integer shadow_id.
    Optionally ensures IDs are contiguous [0..N-1].

    Parameters
    ----------
    dir_path : str
        Directory containing shadow_*.pkl files.
    n_jobs : int
        Joblib parallelism. -1 uses all cores.
    backend : str
        Joblib backend, e.g., "loky" (processes) or "threading".
    check_contiguous : bool
        If True, validates that IDs are contiguous [0..N-1].

    Returns
    -------
    shadows : Dict[int, Any]
        Mapping from shadow_id -> unpickled object (expected keys: args, membership_mask, loss_records).
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Not a directory: {dir_path}")

    paths = sorted(glob.glob(os.path.join(dir_path, "shadow_*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No 'shadow_*.pkl' files found under: {dir_path}")

    # Parse IDs deterministically up front (fails fast on bad names / duplicates)
    id_path_pairs: List[Tuple[int, str]] = []
    seen = set()
    for p in paths:
        fname = os.path.basename(p)
        match = re.match(r"shadow_(\d+)\.pkl$", fname)
        if not match:
            raise ValueError(f"Unexpected filename format: {fname}")
        shadow_id = int(match.group(1))
        if shadow_id in seen:
            raise ValueError(f"Duplicate shadow ID found: {shadow_id}")
        seen.add(shadow_id)
        id_path_pairs.append((shadow_id, p))

    def log(msg: str) -> None:
        if log_fn is not None:
            log_fn(f"{log_prefix} {msg}")
        else:
            print(f"{log_prefix} {msg}")

    selected_shadow_ids = None if shadow_ids is None else {int(sid) for sid in shadow_ids}
    if selected_shadow_ids is not None:
        id_path_pairs = [(sid, path) for sid, path in id_path_pairs if sid in selected_shadow_ids]
        missing_ids = sorted(selected_shadow_ids - {sid for sid, _ in id_path_pairs})
        if missing_ids:
            raise FileNotFoundError(f"Requested shadow ids were not found under {dir_path}: {missing_ids}")

    if max_shadows is not None:
        limit = max(1, int(max_shadows))
        id_path_pairs = id_path_pairs[:limit]

    if not id_path_pairs:
        raise FileNotFoundError(f"No selected 'shadow_*.pkl' files found under: {dir_path}")

    total_bytes = sum(os.path.getsize(path) for _, path in id_path_pairs)
    requested_jobs = len(id_path_pairs) if n_jobs == -1 else int(max(1, n_jobs))
    effective_jobs = min(requested_jobs, len(id_path_pairs))
    effective_batch = int(batch_size) if batch_size is not None else effective_jobs
    effective_batch = max(1, min(effective_batch, len(id_path_pairs)))
    progress_mod = max(1, int(progress_every))
    started_at = time.time()
    log(
        "discovered "
        f"{len(id_path_pairs)} shadow pickles under {dir_path} "
        f"(total={_format_gib(total_bytes)}, backend={backend}, jobs={effective_jobs}, batch_size={effective_batch})"
    )

    shadows: Dict[int, Any] = {}
    if effective_jobs <= 1:
        for loaded_count, (sid, path) in enumerate(id_path_pairs, start=1):
            _, obj = _load_one_shadow(sid, path)
            shadows[sid] = obj
            if loaded_count == 1 or loaded_count == len(id_path_pairs) or loaded_count % progress_mod == 0:
                elapsed_min = (time.time() - started_at) / 60.0
                rss_gib = _rss_gib()
                rss_text = "unknown" if rss_gib is None else f"{rss_gib:.2f} GiB"
                log(
                    f"loaded {loaded_count}/{len(id_path_pairs)} "
                    f"({os.path.basename(path)}, elapsed={elapsed_min:.1f} min, rss={rss_text})"
                )
    else:
        for start in range(0, len(id_path_pairs), effective_batch):
            batch = id_path_pairs[start : start + effective_batch]
            results = Parallel(
                n_jobs=min(effective_jobs, len(batch)),
                backend=backend,
                verbose=0,
            )(delayed(_load_one_shadow)(sid, path) for sid, path in batch)
            for sid, obj in results:
                shadows[sid] = obj
            loaded_count = len(shadows)
            elapsed_min = (time.time() - started_at) / 60.0
            rss_gib = _rss_gib()
            rss_text = "unknown" if rss_gib is None else f"{rss_gib:.2f} GiB"
            batch_desc = ", ".join(os.path.basename(path) for _, path in batch[:2])
            if len(batch) > 2:
                batch_desc += ", ..."
            log(
                f"loaded batch {start // effective_batch + 1}: {loaded_count}/{len(id_path_pairs)} "
                f"(examples={batch_desc}, elapsed={elapsed_min:.1f} min, rss={rss_text})"
            )

    # Optional contiguity check
    if check_contiguous:
        ids = sorted(shadows.keys())
        if ids != list(range(len(ids))):
            raise ValueError(f"Shadow IDs must be contiguous from 0..N-1, found: {ids}")

    # Light schema checks (soft, not failing)
    required_keys = {"args", "membership_mask", "loss_records"}
    for sid, obj in shadows.items():
        if not hasattr(obj, "keys"):
            raise TypeError(f"shadow_{sid}.pkl does not contain a dict-like object.")
        missing = required_keys - set(obj.keys())
        if missing:
            raise KeyError(f"shadow_{sid}.pkl missing keys: {missing}")

    return shadows


def load_reference_pickle(file_path: str):
    """
    Load a reference model pickle file.

    Returns
    -------
    ref : Any
        Unpickled object (expected keys: args, loss_records).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Reference pickle file not found: {file_path}")

    with open(file_path, "rb") as f:
        ref = pickle.load(f)

    # Light schema checks (soft, not failing)
    required_keys = {"args", "loss_records"}
    missing = required_keys - set(ref.keys())
    if missing:
        raise KeyError(f"Reference pickle missing keys: {missing}")

    return ref
