from typing import Dict, Optional, Sequence, Tuple
import math
import torch
from sklearn.covariance import OAS
import numpy as np


def _chol_with_jitter(
    Sigma: torch.Tensor,
    ridge: float,
    max_retries: int = 3,
    diag_atol: float = 1e-3,
) -> Tuple[torch.Tensor, float, int]:
    """
    Cholesky with automatic jitter escalation and automatic diagonal fast-path.

    Returns:
        L          : Cholesky factor (lower-triangular)
        used_ridge : final jitter used
        retries    : number of jitter escalations performed
    """
    D = Sigma.size(0)
    device = Sigma.device
    dtype = Sigma.dtype

    # --- Detect if Sigma is (numerically) diagonal ---
    # Build a diagonal-only matrix and compare.
    # If diag_atol == 0.0 we require exact diagonality.
    diag = torch.diagonal(Sigma)
    Sigma_diag_only = torch.diag(diag)

    if diag_atol == 0.0:
        is_diagonal = torch.equal(Sigma, Sigma_diag_only)
    else:
        diff_max = (Sigma - Sigma_diag_only).abs().max()
        is_diagonal = bool(diff_max <= diag_atol)

    # --- Fast path for diagonal matrices ---
    if is_diagonal:
        jitter = ridge
        for r in range(max_retries + 1):
            diag_perturbed = diag + jitter
            # Need strictly positive entries for sqrt
            if torch.all(diag_perturbed > 0.0):
                L_diag = torch.sqrt(diag_perturbed)
                L = torch.diag(L_diag)
                return L, float(jitter), r
            jitter *= 10.0

        # Final attempt: clip to be PD on the diagonal
        diag_clipped = torch.clamp(diag, min=1e-8)
        L_diag = torch.sqrt(diag_clipped)
        L = torch.diag(L_diag)
        return L, float(jitter), max_retries + 1

    # --- Generic dense-matrix path ---
    jitter = ridge
    I = torch.eye(D, dtype=dtype, device=device)
    for r in range(max_retries + 1):
        try:
            L = torch.linalg.cholesky(Sigma + jitter * I)
            return L, float(jitter), r
        except Exception:
            jitter *= 10.0

    # Final attempt: force PD via eigenvalue clipping
    evals, evecs = torch.linalg.eigh(Sigma)
    evals_clipped = torch.clamp(evals, min=1e-8)
    Sigma_pd = (evecs @ torch.diag(evals_clipped) @ evecs.T).contiguous()
    L = torch.linalg.cholesky(Sigma_pd)
    return L, float(jitter), max_retries + 1


def _chol_diag_with_jitter(
    diag: torch.Tensor,
    ridge: float,
    max_retries: int = 3,
) -> Tuple[torch.Tensor, float, int]:
    """
    Cholesky for diagonal covariance: returns L_diag s.t. Sigma = diag(L_diag^2).
    """
    jitter = ridge
    for r in range(max_retries + 1):
        diag_perturbed = diag + jitter
        if torch.all(diag_perturbed > 0.0):
            return torch.sqrt(diag_perturbed), float(jitter), r
        jitter *= 10.0

    diag_clipped = torch.clamp(diag, min=1e-8)
    return torch.sqrt(diag_clipped), float(jitter), max_retries + 1


def _covariance_by_estimator(
    X: torch.Tensor,  # [n, D]
    covariance_estimator="oas",  # 'oas' (default) or 'empirical'
    eps_sigma: float = 1e-6,
) -> Tuple[torch.Tensor, float]:
    """
    Fit covariance matrix.
    Parameters
    ----------
    X : [n, D] torch.Tensor
    covariance_estimator : 'oas', 'empirical', 'diagonal', 'univariate', 'univariate_fixed', or 'univariate_alt'
    Returns
    -------
    Sigma : [D, D] torch.Tensor
    alpha : float (shrinkage parameter; 0.0 if empirical)
    """

    finite_rows = torch.isfinite(X).all(dim=1)
    X = X[finite_rows]
    if X.size(0) == 0:
        raise ValueError("Covariance estimation received no finite samples.")

    if covariance_estimator == "oas":
        oas = OAS()
        oas.fit(X.cpu().numpy())
        Sigma = oas.covariance_.astype(np.float64, copy=False)
        alpha = float(getattr(oas, "shrinkage_", getattr(oas, "alpha_", 0.0)))
        Sigma = torch.from_numpy(Sigma.copy())
    elif covariance_estimator == "empirical":
        Xc = X - X.mean(dim=0, keepdim=True)
        # Use a safe denominator for n=1 (one IN/OUT sample setting).
        Sigma = (Xc.T @ Xc) / max(X.size(0) - 1, 1)
        alpha = 0.0
    elif covariance_estimator == "diagonal":
        pooled = np.var(X.cpu().numpy(), axis=0).astype(np.float64) + 1e-6
        Sigma = np.diag(pooled).astype(np.float64)
        Sigma = torch.from_numpy(Sigma)
        alpha = 0.0
    elif covariance_estimator == "univariate":
        pooled = np.var(X.cpu().numpy(), axis=0).astype(np.float64) + 1e-6
        avg_var = float(np.mean(pooled))
        Sigma = np.diag(np.full((X.size(1),), avg_var, dtype=np.float64))
        Sigma = torch.from_numpy(Sigma)
        alpha = 0.0
    elif covariance_estimator == "univariate_fixed":
        pooled = np.var(X.cpu().numpy(), axis=0).astype(np.float64) + 1e-6
        avg_var = float(np.mean(pooled))
        Sigma = np.diag(np.full((X.size(1),), avg_var, dtype=np.float64))
        Sigma = torch.from_numpy(Sigma)
        alpha = 0.0
    else:
        raise ValueError(f"Unknown covariance estimator: {covariance_estimator}")

    # Sigma += eps_sigma * torch.eye(Sigma.size(0), dtype=Sigma.dtype)
    return Sigma, alpha


def compute_covariance(
    X_in: torch.Tensor,
    X_out: torch.Tensor,
    covariance_estimator="oas",  # 'oas' (default) or 'empirical'
    shared_covariance: bool = True,
    ridge: float = 1e-6,
    max_retries: int = 3,
    X_star: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Fit class means and covariance matrices.

    Parameters
    ----------
    X_in  : [n_in, D] torch.Tensor
    X_out : [n_out, D] torch.Tensor
    covariance_estimator : 'oas', 'empirical', 'diagonal', 'univariate', 'univariate_fixed', or 'univariate_alt'
    shared_covariance : if True, fit a single shared covariance; else fit separate covariances

    Returns
    -------
    mu_in  : [D] torch.Tensor
    mu_out : [D] torch.Tensor
    L_in  : [D, D] or [D] torch.Tensor (Cholesky of Sigma_in; 1-D for diagonal)
    L_out : [D, D] or [D] torch.Tensor (Cholesky of Sigma_out; same as L_in if shared_covariance)
    alpha_in  : float (shrinkage parameter; 0.0 if empirical)
    alpha_out : float (shrinkage parameter; 0.0 if empirical; same as alpha_in if shared_covariance)
    """
    finite_in = torch.isfinite(X_in).all(dim=1)
    finite_out = torch.isfinite(X_out).all(dim=1)
    X_in = X_in[finite_in]
    X_out = X_out[finite_out]
    if X_in.size(0) == 0 or X_out.size(0) == 0:
        raise ValueError("Covariance estimation requires at least one finite IN and OUT sample.")
    if X_star is not None and not bool(torch.isfinite(X_star).all()):
        raise ValueError("Target feature vector contains non-finite values.")

    mu_in = X_in.mean(dim=0)
    mu_out = X_out.mean(dim=0)

    if covariance_estimator == "univariate_alt":
        # Univariate-alt: per-feature mean, with two scalar variances (first/second half).
        # Non-constant mask is computed across IN/OUT/target together.
        D = X_in.size(1)
        eps = 1e-6
        mid = D // 2
        X_in_d = X_in.double()
        X_out_d = X_out.double()
        mu_in_d = mu_in.double()
        mu_out_d = mu_out.double()

        def _avg_var_for_slice(Xc, sl):
            return float(Xc[:, sl].pow(2).mean().item()) + eps

        if shared_covariance:
            Xc = torch.cat([(X_in_d - mu_in_d), (X_out_d - mu_out_d)], dim=0)
            avg_var_1 = _avg_var_for_slice(Xc, slice(0, mid))
            avg_var_2 = _avg_var_for_slice(Xc, slice(mid, D))
            var_diag = torch.full((D,), avg_var_2, dtype=torch.float64)
            var_diag[:mid] = avg_var_1
            L_diag, used_jitter, retries = _chol_diag_with_jitter(
                var_diag, ridge=ridge, max_retries=max_retries
            )
            return (
                mu_in_d,
                mu_out_d,
                L_diag.to(dtype=torch.float64),
                L_diag.to(dtype=torch.float64),
                0.0,
                0.0,
            )
        else:
            Xc_in = X_in_d - mu_in_d
            Xc_out = X_out_d - mu_out_d
            avg_var_in_1 = _avg_var_for_slice(Xc_in, slice(0, mid))
            avg_var_in_2 = _avg_var_for_slice(Xc_in, slice(mid, D))
            avg_var_out_1 = _avg_var_for_slice(Xc_out, slice(0, mid))
            avg_var_out_2 = _avg_var_for_slice(Xc_out, slice(mid, D))
            var_diag_in = torch.full((D,), avg_var_in_2, dtype=torch.float64)
            var_diag_out = torch.full((D,), avg_var_out_2, dtype=torch.float64)
            var_diag_in[:mid] = avg_var_in_1
            var_diag_out[:mid] = avg_var_out_1
            L_in_diag, used_jitter_in, retries_in = _chol_diag_with_jitter(
                var_diag_in, ridge=ridge, max_retries=max_retries
            )
            L_out_diag, used_jitter_out, retries_out = _chol_diag_with_jitter(
                var_diag_out, ridge=ridge, max_retries=max_retries
            )
            return (
                mu_in_d,
                mu_out_d,
                L_in_diag.to(dtype=torch.float64),
                L_out_diag.to(dtype=torch.float64),
                0.0,
                0.0,
            )

    cov_estimator = covariance_estimator
    if covariance_estimator == "univariate_fixed":
        # Univariate-fixed: force scalar mean across features, then use univariate variance.
        mu_in_scalar = X_in.mean()
        mu_out_scalar = X_out.mean()
        mu_in = mu_in_scalar.expand_as(mu_in)
        mu_out = mu_out_scalar.expand_as(mu_out)
        cov_estimator = "univariate"
    diag_only = cov_estimator in ("diagonal", "univariate")
    if shared_covariance:
        Xc_in = (X_in - mu_in).double()
        Xc_out = (X_out - mu_out).double()
        Xc = torch.cat([Xc_in, Xc_out], dim=0)  # [n, D]
        n, D = Xc.shape
        if diag_only:
            var_diag = Xc.var(dim=0, unbiased=False) + 1e-6
            if cov_estimator == "univariate":
                avg_var = float(var_diag.mean().item())
                var_diag = torch.full((D,), avg_var, dtype=torch.float64)
            L_diag, used_jitter, retries = _chol_diag_with_jitter(
                var_diag, ridge=ridge, max_retries=max_retries
            )
            return (
                mu_in.to(dtype=torch.float64),
                mu_out.to(dtype=torch.float64),
                L_diag.to(dtype=torch.float64),
                L_diag.to(dtype=torch.float64),
                0.0,
                0.0,
            )
        Sigma, alpha = _covariance_by_estimator(Xc, covariance_estimator=cov_estimator)
        L, used_jitter, retries = _chol_with_jitter(
            Sigma, ridge=ridge, max_retries=max_retries
        )
        return (
            mu_in.to(dtype=torch.float64),
            mu_out.to(dtype=torch.float64),
            L.to(dtype=torch.float64),
            L.to(dtype=torch.float64),
            alpha,
            alpha,
        )
    else:
        if diag_only:
            Xc_in = (X_in - mu_in).double()
            Xc_out = (X_out - mu_out).double()
            var_diag_in = Xc_in.var(dim=0, unbiased=False) + 1e-6
            var_diag_out = Xc_out.var(dim=0, unbiased=False) + 1e-6
            if cov_estimator == "univariate":
                avg_var_in = float(var_diag_in.mean().item())
                avg_var_out = float(var_diag_out.mean().item())
                D = var_diag_in.numel()
                var_diag_in = torch.full((D,), avg_var_in, dtype=torch.float64)
                var_diag_out = torch.full((D,), avg_var_out, dtype=torch.float64)
            L_in_diag, used_jitter_in, retries_in = _chol_diag_with_jitter(
                var_diag_in, ridge=ridge, max_retries=max_retries
            )
            L_out_diag, used_jitter_out, retries_out = _chol_diag_with_jitter(
                var_diag_out, ridge=ridge, max_retries=max_retries
            )
            return (
                mu_in.to(dtype=torch.float64),
                mu_out.to(dtype=torch.float64),
                L_in_diag.to(dtype=torch.float64),
                L_out_diag.to(dtype=torch.float64),
                0.0,
                0.0,
            )
        Sigma_in, alpha_in = _covariance_by_estimator(
            X_in, covariance_estimator=cov_estimator
        )
        Sigma_out, alpha_out = _covariance_by_estimator(
            X_out, covariance_estimator=cov_estimator
        )
        L_in, used_jitter_in, retries_in = _chol_with_jitter(
            Sigma_in, ridge=ridge, max_retries=max_retries
        )
        L_out, used_jitter_out, retries_out = _chol_with_jitter(
            Sigma_out, ridge=ridge, max_retries=max_retries
        )
        return (
            mu_in.to(dtype=torch.float64),
            mu_out.to(dtype=torch.float64),
            L_in.to(dtype=torch.float64),
            L_out.to(dtype=torch.float64),
            alpha_in,
            alpha_out,
        )


def _logpdf_mvn_chol(
    x: torch.Tensor, mu: torch.Tensor, L: torch.Tensor
) -> torch.Tensor:
    """
    Log N(x | mu, Sigma) given lower-triangular Cholesky L s.t. Sigma = L @ L^T.
    If L is 1-D, treat it as diagonal entries of L.
    """
    D = x.numel()
    xc = (x - mu).reshape(-1)
    if L.ndim == 1:
        inv = xc / L
        quad = (inv * inv).sum()
        logdet = 2.0 * torch.log(L).sum()
        return -0.5 * (D * torch.log(torch.tensor(2.0 * torch.pi)) + logdet + quad)
    # Solve L * y = xc  -> y = L^{-1} xc
    y = torch.cholesky_solve(
        xc[:, None], L
    )  # solves (L L^T) y = xc, but we need only quadratic
    quad = (xc[:, None] * y).sum()
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    return -0.5 * (D * torch.log(torch.tensor(2.0 * torch.pi)) + logdet + quad)


def _collect_in_out(
    V: torch.Tensor,
    valid: torch.Tensor,
    M: torch.Tensor,
    t: int,
    i: int,
    n_min: int,
    all_samples: bool = False,
    mask_others: Optional[torch.Tensor] = None,
    num_in_models: Optional[int] = None,
    num_out_models: Optional[int] = None,
):
    """
    Helper to collect IN/OUT samples for target t and sample i.

    By default this implementation intentionally uses exactly one IN shadow and one
    OUT shadow per evaluation (t, i). When num_in_models / num_out_models are set,
    it deterministically selects that many per class.
    """
    if not bool(valid[t, i]):
        return None

    if mask_others is None:
        mask_others = torch.ones((V.shape[0],), dtype=torch.bool, device=valid.device)
        mask_others[t] = False
    valid_i = valid[:, i] & mask_others
    if valid_i.sum().item() < 2 * n_min:
        return None

    is_in = M[:, i] & valid_i
    is_out = ~M[:, i] & valid_i
    n_in, n_out = int(is_in.sum()), int(is_out.sum())
    selected_n_in = 1 if num_in_models is None else int(num_in_models)
    selected_n_out = 1 if num_out_models is None else int(num_out_models)
    if selected_n_in <= 0 or selected_n_out <= 0:
        raise ValueError("num_in_models and num_out_models must be positive when provided.")

    required_n_in = max(int(n_min), selected_n_in)
    required_n_out = max(int(n_min), selected_n_out)
    if n_in < required_n_in or n_out < required_n_out:
        return None

    in_ids = torch.nonzero(is_in, as_tuple=False).reshape(-1)
    out_ids = torch.nonzero(is_out, as_tuple=False).reshape(-1)
    # Deterministic subset choice; rotates with (target, sample) to avoid always
    # picking the same shadows when multiple candidates exist.
    in_start = int((t + i) % in_ids.numel())
    out_start = int((t + i) % out_ids.numel())
    in_ids = torch.roll(in_ids, shifts=-in_start, dims=0)[:selected_n_in]
    out_ids = torch.roll(out_ids, shifts=-out_start, dims=0)[:selected_n_out]

    if all_samples:
        X_in, X_out = V[in_ids, :, :], V[out_ids, :, :]
    else:
        X_in = V[in_ids, i, :]
        X_out = V[out_ids, i, :]
    return X_in, X_out, int(in_ids.numel()), int(out_ids.numel())


def compute_lira_for_target(
    V: torch.Tensor,  # [S, N, D]
    valid: torch.Tensor,  # [S, N] bool
    M: torch.Tensor,  # [S, N] bool
    target_idx: int,
    *,
    n_min: int = 1,
    ridge: float = 1e-3,
    max_retries: int = 10,
    covariance_estimator: str = "oas",  # also supports 'pair_diff'
    shared_covariance: bool = True,
    num_in_models: Optional[int] = None,
    num_out_models: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Compact version of Leave-one-out LiRA for a single target shadow."""
    S, N, D = V.shape
    t = int(target_idx)
    if not (0 <= t < S):
        raise IndexError(f"target_idx {t} out of range [0, {S-1}]")
    selected_num_in = 1 if num_in_models is None else int(num_in_models)
    selected_num_out = 1 if num_out_models is None else int(num_out_models)
    use_single_pair_fastpath = selected_num_in == 1 and selected_num_out == 1

    llr = torch.full((N,), float("nan"), dtype=V.dtype, device=V.device)
    scored = torch.zeros((N,), dtype=torch.bool)
    finite_valid = valid & torch.isfinite(V).all(dim=2)
    mask_others = torch.ones((S,), dtype=torch.bool, device=valid.device)
    mask_others[t] = False

    # Precompute valid indices and deterministic IN/OUT shadow IDs for the
    # legacy single-IN/single-OUT fast path.
    idxs = []
    in_sids = []
    out_sids = []
    if use_single_pair_fastpath:
        for i in range(N):
            if not bool(finite_valid[t, i]):
                continue
            valid_i = finite_valid[:, i] & mask_others
            if valid_i.sum().item() < 2 * n_min:
                continue
            is_in = M[:, i] & valid_i
            is_out = ~M[:, i] & valid_i
            n_in, n_out = int(is_in.sum()), int(is_out.sum())
            if n_in < n_min or n_out < n_min:
                continue
            in_ids = torch.nonzero(is_in, as_tuple=False).reshape(-1)
            out_ids = torch.nonzero(is_out, as_tuple=False).reshape(-1)
            in_sid = int(in_ids[(t + i) % in_ids.numel()].item())
            out_sid = int(out_ids[(t + i) % out_ids.numel()].item())
            idxs.append(i)
            in_sids.append(in_sid)
            out_sids.append(out_sid)
    else:
        idxs = [
            i
            for i in range(N)
            if _collect_in_out(
                V,
                finite_valid,
                M,
                t,
                i,
                n_min,
                mask_others=mask_others,
                num_in_models=selected_num_in,
                num_out_models=selected_num_out,
            )
            is not None
        ]

    if not idxs:
        return llr, scored, {}

    idxs_t = torch.as_tensor(idxs, device=V.device, dtype=torch.long)
    in_sids_t = torch.as_tensor(in_sids, device=V.device, dtype=torch.long)
    out_sids_t = torch.as_tensor(out_sids, device=V.device, dtype=torch.long)

    if covariance_estimator == "pair_diff":
        if D < 2:
            return llr, scored, {}

        d_chosen = int(D // 2)
        d_rejected = int(D - d_chosen)
        chosen_llr = torch.full((N,), float("nan"), dtype=llr.dtype, device=llr.device)
        rejected_llr = torch.full((N,), float("nan"), dtype=llr.dtype, device=llr.device)

        def _tail_valid_mask_1d(x_1d: torch.Tensor) -> torch.Tensor:
            """
            For half-pad vectors, treat trailing zeros as padding.
            Scalar halves (e.g. average+both) are always kept.
            """
            x_1d = x_1d.reshape(-1)
            if x_1d.numel() <= 1:
                return torch.isfinite(x_1d)
            nz = (x_1d != 0) & torch.isfinite(x_1d)
            if not bool(nz.any()):
                return torch.zeros_like(nz, dtype=torch.bool)
            last = torch.nonzero(nz, as_tuple=False)[-1, 0]
            idx = torch.arange(x_1d.numel(), device=x_1d.device)
            return idx <= last

        def _half_llr(
            x_in_half: torch.Tensor,
            x_out_half: torch.Tensor,
            x_star_half: torch.Tensor,
            keep_mask: torch.Tensor,
        ) -> Optional[torch.Tensor]:
            keep_mask = (
                keep_mask
                & torch.isfinite(x_star_half)
                & torch.isfinite(x_in_half).all(dim=0)
                & torch.isfinite(x_out_half).all(dim=0)
            )
            if not bool(keep_mask.any()):
                return None
            X_in_h = x_in_half[:, keep_mask]
            X_out_h = x_out_half[:, keep_mask]
            x_star_h = x_star_half[keep_mask]
            mu_in_h, mu_out_h, L_in_h, L_out_h, _, _ = compute_covariance(
                X_in_h,
                X_out_h,
                covariance_estimator="univariate",
                shared_covariance=shared_covariance,
                ridge=ridge,
                max_retries=max_retries,
                X_star=x_star_h,
            )
            return (
                _logpdf_mvn_chol(x_star_h, mu_in_h, L_in_h)
                - _logpdf_mvn_chol(x_star_h, mu_out_h, L_out_h)
            ).to(llr.dtype)

        iterator = zip(idxs, in_sids, out_sids) if use_single_pair_fastpath else ((i, None, None) for i in idxs)
        for i, in_sid, out_sid in iterator:
            if use_single_pair_fastpath:
                X_in = V[in_sid, i, :].unsqueeze(0)
                X_out = V[out_sid, i, :].unsqueeze(0)
            else:
                collected = _collect_in_out(
                    V,
                    finite_valid,
                    M,
                    t,
                    i,
                    n_min,
                    mask_others=mask_others,
                    num_in_models=selected_num_in,
                    num_out_models=selected_num_out,
                )
                if collected is None:
                    continue
                X_in, X_out, _, _ = collected
            x_star = V[t, i, :]

            x_in_c, x_in_r = X_in[:, :d_chosen], X_in[:, d_chosen:]
            x_out_c, x_out_r = X_out[:, :d_chosen], X_out[:, d_chosen:]
            x_star_c, x_star_r = x_star[:d_chosen], x_star[d_chosen:]

            keep_c = (
                _tail_valid_mask_1d(x_in_c[0])
                & _tail_valid_mask_1d(x_out_c[0])
                & _tail_valid_mask_1d(x_star_c)
            )
            keep_r = (
                _tail_valid_mask_1d(x_in_r[0])
                & _tail_valid_mask_1d(x_out_r[0])
                & _tail_valid_mask_1d(x_star_r)
            )

            llr_c = _half_llr(x_in_c, x_out_c, x_star_c, keep_c)
            llr_r = _half_llr(x_in_r, x_out_r, x_star_r, keep_r)
            if llr_c is None or llr_r is None:
                continue

            chosen_llr[i] = llr_c
            rejected_llr[i] = llr_r
            llr[i] = llr_c - llr_r
            scored[i] = True

        return llr, scored, {"lira_pair_chosen": chosen_llr, "lira_pair_rejected": rejected_llr}

    # Fast vectorized path for diagonal/univariate estimators.
    fast_cov = {"diagonal", "univariate", "univariate_fixed"}
    if covariance_estimator in fast_cov and use_single_pair_fastpath:
        X_in = V[in_sids_t, idxs_t, :].to(torch.float64)
        X_out = V[out_sids_t, idxs_t, :].to(torch.float64)
        x_star = V[t, idxs_t, :].to(torch.float64)
        finite_rows = (
            torch.isfinite(X_in).all(dim=1)
            & torch.isfinite(X_out).all(dim=1)
            & torch.isfinite(x_star).all(dim=1)
        )
        if not bool(finite_rows.any()):
            return llr, scored, {}
        X_in = X_in[finite_rows]
        X_out = X_out[finite_rows]
        x_star = x_star[finite_rows]
        idxs_t = idxs_t[finite_rows]

        cov_mode = covariance_estimator
        if covariance_estimator == "univariate_fixed":
            mu_in_scalar = X_in.mean(dim=1, keepdim=True)
            mu_out_scalar = X_out.mean(dim=1, keepdim=True)
            mu_in = mu_in_scalar.expand_as(X_in)
            mu_out = mu_out_scalar.expand_as(X_out)
            cov_mode = "univariate"
        else:
            mu_in = X_in
            mu_out = X_out

        if shared_covariance:
            Xc = torch.stack([X_in - mu_in, X_out - mu_out], dim=1)  # [K, 2, D]
            var_diag = Xc.var(dim=1, unbiased=False) + 1e-6
            if cov_mode == "univariate":
                avg_var = var_diag.mean(dim=1, keepdim=True)
                var_diag = avg_var.expand_as(var_diag)
            L_in = torch.sqrt(var_diag + ridge)
            L_out = L_in
        else:
            # With one IN/OUT sample, per-class variance is zero (plus eps).
            var_in = torch.full_like(X_in, 1e-6)
            var_out = torch.full_like(X_out, 1e-6)
            if cov_mode == "univariate":
                var_in = var_in.mean(dim=1, keepdim=True).expand_as(var_in)
                var_out = var_out.mean(dim=1, keepdim=True).expand_as(var_out)
            L_in = torch.sqrt(var_in + ridge)
            L_out = torch.sqrt(var_out + ridge)

        # Diagonal log-pdf in batch.
        const = float(D) * math.log(2.0 * math.pi)
        const_t = torch.tensor(const, dtype=torch.float64, device=V.device)
        inv_in = (x_star - mu_in) / L_in
        inv_out = (x_star - mu_out) / L_out
        quad_in = (inv_in * inv_in).sum(dim=1)
        quad_out = (inv_out * inv_out).sum(dim=1)
        logdet_in = 2.0 * torch.log(L_in).sum(dim=1)
        logdet_out = 2.0 * torch.log(L_out).sum(dim=1)
        logp_in = -0.5 * (const_t + logdet_in + quad_in)
        logp_out = -0.5 * (const_t + logdet_out + quad_out)
        llr_vals = (logp_in - logp_out).to(llr.dtype)

        llr[idxs_t] = llr_vals
        scored[idxs_t] = True
        return llr, scored, {}

    # Fallback: exact per-sample covariance (oas/empirical/univariate_alt).
    iterator = zip(idxs, in_sids, out_sids) if use_single_pair_fastpath else ((i, None, None) for i in idxs)
    for i, in_sid, out_sid in iterator:
        if use_single_pair_fastpath:
            X_in = V[in_sid, i, :].unsqueeze(0)
            X_out = V[out_sid, i, :].unsqueeze(0)
        else:
            collected = _collect_in_out(
                V,
                finite_valid,
                M,
                t,
                i,
                n_min,
                mask_others=mask_others,
                num_in_models=selected_num_in,
                num_out_models=selected_num_out,
            )
            if collected is None:
                continue
            X_in, X_out, _, _ = collected
        x_star = V[t, i, :]
        if not bool(torch.isfinite(x_star).all()):
            continue
        try:
            mu_in, mu_out, L_in, L_out, _, _ = compute_covariance(
                X_in,
                X_out,
                covariance_estimator=covariance_estimator,
                shared_covariance=shared_covariance,
                ridge=ridge,
                max_retries=max_retries,
                X_star=x_star,
            )
        except ValueError:
            continue
        llr[i] = (
            _logpdf_mvn_chol(x_star, mu_in, L_in)
            - _logpdf_mvn_chol(x_star, mu_out, L_out)
        ).double()
        scored[i] = True

    return llr, scored, {}


def compute_rmia_for_target(
    V: torch.Tensor,  # [S, N, D]
    valid: torch.Tensor,
    M: torch.Tensor,
    target_idx: int,
    *,
    n_min: int = 1,
    mode: str = "univariate",  # 'univariate' (default), 'multivariate', 'multivariate_exact', or 'multivariate_pair'
    version: str = "standard",  # 'standard' or 'info'
    log_gamma: float = 1.0,
    offline : bool = False,
    num_in_models: Optional[int] = None,
    num_out_models: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Compact version of Leave-one-out RMIA for a single target shadow."""
    if mode not in {"univariate", "multivariate", "multivariate_exact", "multivariate_pair"}:
        raise ValueError(f"Unknown mode: {mode}")

    finite_valid = valid & torch.isfinite(V).all(dim=2)
    if mode == "univariate":
        V = torch.nanmean(V, dim=-1, keepdim=True)
    exact_multivariate = mode == "multivariate_exact"
    pair_multivariate = mode == "multivariate_pair"

    S, N, D = V.shape
    t = int(target_idx)
    if not (0 <= t < S):
        raise IndexError(f"target_idx {t} out of range [0, {S-1}]")
    if pair_multivariate and D < 2:
        # Pair mode requires a chosen/rejected split, which needs at least 2 dims.
        return (
            torch.full((N,), float("nan"), dtype=V.dtype, device=V.device),
            torch.zeros((N,), dtype=torch.bool, device=V.device),
            {},
        )
    def _tail_valid_mask_1d(x_1d: torch.Tensor) -> torch.Tensor:
        """
        For half_pad_concat, padding is trailing zeros. Mark dims up to the last non-zero as valid.
        If all zeros, returns all-False.
        """
        x_1d = x_1d.reshape(-1)
        nz = (x_1d != 0) & torch.isfinite(x_1d)
        if not bool(nz.any()):
            return torch.zeros_like(nz, dtype=torch.bool)
        last = torch.nonzero(nz, as_tuple=False)[-1, 0]
        idx = torch.arange(x_1d.numel(), device=x_1d.device)
        return idx <= last

    def _logmeanexp_ignore_tailpad(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-dimension log-mean-exp, treating trailing zeros in each row as padding/missing.

        Args:
            X: [K, D] matrix.
        Returns:
            logp:   [D] where logp[d] = log(mean_k exp(X[k,d])) over k where (k,d) is valid.
            counts: [D] number of valid rows per dimension.
        """
        K, Dloc = X.shape
        idx = torch.arange(Dloc, device=X.device).view(1, -1)  # [1, D]
        # last_nonzero[k] in [-1, D-1]
        nz = X != 0
        masked_idx = torch.where(
            nz,
            idx.expand(K, Dloc),
            torch.full((K, Dloc), -1, device=X.device, dtype=idx.dtype),
        )
        last = masked_idx.max(dim=1).values.view(-1, 1)  # [K, 1]
        valid_kd = idx <= last  # [K, D]
        counts = valid_kd.sum(dim=0)  # [D]

        X64 = X.to(torch.float64)
        X_masked = X64.masked_fill(~valid_kd, float("-inf"))
        lse = torch.logsumexp(X_masked, dim=0)  # [D]
        # Avoid -inf - log(0); caller should drop dims with counts==0.
        counts64 = counts.to(torch.float64).clamp_min(1.0)
        logp = lse - counts64.log()
        return logp.to(dtype=X.dtype), counts

    llr = torch.full((N,), float("nan"), dtype=V.dtype, device=V.device)
    scored = torch.zeros((N,), dtype=torch.bool)
    mask_others = torch.ones((S,), dtype=torch.bool, device=valid.device)
    mask_others[t] = False

    # Precompute shared log ratios and cache per-sample stats
    lratio_z, logp_z, ids_z = [], [], []
    lratio_z_c, lratio_z_r = [], []
    logp_cache = {}
    for i in range(N):
        result = _collect_in_out(
            V,
            finite_valid,
            M,
            t,
            i,
            n_min,
            mask_others=mask_others,
            num_in_models=num_in_models,
            num_out_models=num_out_models,
        )
        if result is None:
            continue
        X_in, X_out, _, _ = result
        if offline:
            X = X_out
        else:
            X = torch.cat([X_in, X_out], dim=0)

        if pair_multivariate:
            # Split chosen/rejected halves and ignore half_pad_concat trailing padding per half.
            d_chosen = int(D // 2)
            d_rejected = int(D - d_chosen)
            Xc = X[:, :d_chosen]
            Xr = X[:, d_chosen:]

            logp_c, cnt_c = _logmeanexp_ignore_tailpad(Xc)
            logp_r, cnt_r = _logmeanexp_ignore_tailpad(Xr)

            vt_c = V[t, i, :d_chosen]
            vt_r = V[t, i, d_chosen:]
            keep_c = (cnt_c > 0) & _tail_valid_mask_1d(vt_c)
            keep_r = (cnt_r > 0) & _tail_valid_mask_1d(vt_r)
            if not bool(keep_c.any()) or not bool(keep_r.any()):
                continue

            # Scalar exact-style lratios for each half.
            lratio_z_c.append((vt_c[keep_c] - logp_c[keep_c]).sum())
            lratio_z_r.append((vt_r[keep_r] - logp_r[keep_r]).sum())
            ids_z.append(i)
            continue

        logp = X.logsumexp(dim=0) - np.log(X.shape[0])
        logp_cache[i] = logp
        logp_z.append(logp)
        if exact_multivariate:
            # Paper-faithful multivariate RMIA uses one LR per sample:
            # LR(x,z) = [P(x|theta)/P(x)] / [P(z|theta)/P(z)].
            # We compute this in log-space via summed per-dimension terms.
            lratio_z.append((V[t, i, :] - logp).sum())
        else:
            lratio_z.append(V[t, i, :] - logp)
        ids_z.append(i)

    if pair_multivariate:
        if not lratio_z_c:
            return llr, scored, {}
        ids_z = torch.tensor(ids_z)
        lratio_z_c = torch.stack(lratio_z_c)  # [K]
        lratio_z_r = torch.stack(lratio_z_r)  # [K]
        K = int(lratio_z_c.size(0))
        if K <= 1:
            return llr, scored, {}

        def _exact_standard_scores(lratio_scalar: torch.Tensor) -> torch.Tensor:
            sorted_vals, _ = torch.sort(lratio_scalar)
            thresholds = lratio_scalar - log_gamma
            counts = torch.searchsorted(sorted_vals, thresholds, right=True).to(
                dtype=torch.float64
            )
            if log_gamma <= 0:
                counts = counts - 1.0
            counts = torch.clamp(counts, min=0.0)
            return (counts / float(K - 1)).to(dtype=llr.dtype)

        if version == "standard":
            s_c = _exact_standard_scores(lratio_z_c)
            s_r = _exact_standard_scores(lratio_z_r)
            llr_vals = s_c - s_r
            llr[ids_z] = llr_vals
            scored[ids_z] = True
            return llr, scored, {"rmia_pair_chosen": s_c, "rmia_pair_rejected": s_r}

        if version == "info":
            # Mirror the exact-multivariate info construction, but do it per half.
            # Note: this uses scalar per-sample summaries for each half after dropping padding.
            logp_c = lratio_z_c.new_empty((K,))
            logp_r = lratio_z_r.new_empty((K,))
            logp_theta_c = lratio_z_c.new_empty((K,))
            logp_theta_r = lratio_z_r.new_empty((K,))
            # Reconstruct scalar logp and theta scalars by re-running the per-sample loop cheaply.
            # (We didn't stash them to keep memory down.)
            for j, i in enumerate(ids_z.tolist()):
                result = _collect_in_out(
                    V,
                    finite_valid,
                    M,
                    t,
                    int(i),
                    n_min,
                    mask_others=mask_others,
                    num_in_models=num_in_models,
                    num_out_models=num_out_models,
                )
                if result is None:
                    # Shouldn't happen, but keep tensors finite.
                    logp_c[j] = 0.0
                    logp_r[j] = 0.0
                    logp_theta_c[j] = 0.0
                    logp_theta_r[j] = 0.0
                    continue
                X_in, X_out, _, _ = result
                X = X_out if offline else torch.cat([X_in, X_out], dim=0)
                d_chosen = int(D // 2)
                Xc = X[:, :d_chosen]
                Xr = X[:, d_chosen:]
                lp_c_vec, cnt_c = _logmeanexp_ignore_tailpad(Xc)
                lp_r_vec, cnt_r = _logmeanexp_ignore_tailpad(Xr)
                vt_c = V[t, int(i), :d_chosen]
                vt_r = V[t, int(i), d_chosen:]
                keep_c = (cnt_c > 0) & _tail_valid_mask_1d(vt_c)
                keep_r = (cnt_r > 0) & _tail_valid_mask_1d(vt_r)
                logp_c[j] = lp_c_vec[keep_c].sum()
                logp_r[j] = lp_r_vec[keep_r].sum()
                logp_theta_c[j] = vt_c[keep_c].sum()
                logp_theta_r[j] = vt_r[keep_r].sum()

            def _info_scores(lratio_scalar: torch.Tensor, logp_scalar: torch.Tensor, logp_theta_scalar: torch.Tensor) -> torch.Tensor:
                lp = logp_scalar.to(torch.float64)
                lp = lp - torch.logsumexp(lp, dim=0)
                lpt = logp_theta_scalar.to(torch.float64)
                lpt = lpt - torch.logsumexp(lpt, dim=0)
                kl = (lp.exp() * (lp - lpt)).sum()
                return (-lratio_scalar.to(torch.float64) + kl).to(dtype=llr.dtype)

            s_c = _info_scores(lratio_z_c, logp_c, logp_theta_c)
            s_r = _info_scores(lratio_z_r, logp_r, logp_theta_r)
            llr_vals = s_c - s_r
            llr[ids_z] = llr_vals
            scored[ids_z] = True
            return llr, scored, {"rmia_pair_chosen": s_c, "rmia_pair_rejected": s_r}

        raise ValueError(f"Unknown version: {version}")

    if not lratio_z:
        return llr, scored, {}

    lratio_z = torch.stack(lratio_z)
    logp_z = torch.stack(logp_z)
    ids_z = torch.tensor(ids_z)
    K = int(lratio_z.size(0))
    if K <= 1:
        return llr, scored, {}

    if version == "info":
        if exact_multivariate:
            # Exact multivariate info variant: build a sample-level KL term from
            # scalar log-likelihood summaries (sum over dimensions first).
            logp_z_scalar = logp_z.sum(dim=1)  # [K]
            logp_z_scalar = logp_z_scalar - logp_z_scalar.logsumexp(dim=0)
            logp_z_theta_scalar = V[t, ids_z, :].sum(dim=1)  # [K]
            logp_z_theta_scalar = logp_z_theta_scalar - logp_z_theta_scalar.logsumexp(
                dim=0
            )
            kl_z = (logp_z_scalar.exp() * (logp_z_scalar - logp_z_theta_scalar)).sum()
            llr_vals = -lratio_z + kl_z
        else:
            logsumexp_z = logp_z.logsumexp(dim=0)
            logp_z = logp_z - logsumexp_z
            logp_z_theta = V[t, ids_z, :]
            logp_z_theta = logp_z_theta - logp_z_theta.logsumexp(dim=0, keepdim=True)
            kl_z = (logp_z.exp() * (logp_z - logp_z_theta)).sum(dim=0)
            llr_vals = (-lratio_z + kl_z).mean(dim=1)

        llr_vals = llr_vals.to(llr.dtype)
        llr[ids_z] = llr_vals
        scored[ids_z] = True
        return llr, scored, {}

    if version == "standard":
        if exact_multivariate:
            # Equation (5) in RMIA: Score = Pr_z[ LR(x,z) >= gamma ].
            # In log-space this is: lratio_x - lratio_z >= log_gamma.
            sorted_vals, _ = torch.sort(lratio_z)
            thresholds = lratio_z - log_gamma
            counts = torch.searchsorted(sorted_vals, thresholds, right=True).to(
                dtype=torch.float64
            )
            if log_gamma <= 0:
                counts = counts - 1.0
            counts = torch.clamp(counts, min=0.0)
            llr_vals = counts / float(K - 1)
            llr_vals = llr_vals.to(llr.dtype)
            llr[ids_z] = llr_vals
            scored[ids_z] = True
            return llr, scored, {}

        # Legacy multivariate/univariate standard: block vectorization over i
        lratio_mat = lratio_z  # [K, D]
        D = int(lratio_mat.size(1))
        elem_size = lratio_mat.element_size()
        target_bytes = 1024 * 1024 * 1024  # ~1024MB block target
        block = max(1, int(target_bytes // max(1, K * D * elem_size)))
        denom = float(K - 1)
        for start in range(0, K, block):
            end = min(K, start + block)
            lr_x = lratio_mat[start:end]  # [B, D]
            cmp = (lr_x[:, None, :] - lratio_mat[None, :, :] < log_gamma).to(
                torch.float64
            ).mean(dim=2)  # [B, K]
            row_idx = torch.arange(end - start, device=cmp.device)
            col_idx = row_idx + start
            cmp_sum = cmp.sum(dim=1) - cmp[row_idx, col_idx]
            llr_vals = cmp_sum / denom
            llr_vals = llr_vals.to(llr.dtype)
            llr[ids_z[start:end]] = llr_vals
            scored[ids_z[start:end]] = True
        return llr, scored, {}

    raise ValueError(f"Unknown version: {version}")


def compute_roc_auc(y_true: torch.Tensor, scores: torch.Tensor):
    """
    ROC points and AUROC.
    y_true: [M] int/bool (1=member, 0=non-member)
    scores: [M] float (higher => more IN)
    Returns (fpr [K], tpr [K], auc float)
    """
    y = y_true.to(torch.int32).reshape(-1).cpu()
    s = scores.to(torch.float64).reshape(-1).cpu()
    M = y.numel()
    if M == 0:
        return (torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), float("nan"))

    P = int((y == 1).sum().item())
    N = int((y == 0).sum().item())
    if P == 0 or N == 0:
        # Degenerate case
        return (torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), float("nan"))

    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    s_sorted = s[order]

    # cumulative TP/FP at each prefix
    tp = torch.cumsum((y_sorted == 1).to(torch.int64), dim=0)
    fp = torch.cumsum((y_sorted == 0).to(torch.int64), dim=0)

    # take unique score cutpoints
    distinct = torch.ones_like(s_sorted, dtype=torch.bool)
    distinct[1:] = s_sorted[1:] != s_sorted[:-1]
    idx = torch.nonzero(distinct, as_tuple=False).view(-1)

    # prepend (0,0), append (1,1)
    tpr = torch.cat(
        [torch.tensor([0.0]), tp[idx].to(torch.float64) / P, torch.tensor([1.0])]
    )
    fpr = torch.cat(
        [torch.tensor([0.0]), fp[idx].to(torch.float64) / N, torch.tensor([1.0])]
    )

    # trapezoidal AUC
    auc = torch.trapz(tpr, fpr).item()
    return fpr.to(torch.float64), tpr.to(torch.float64), float(auc)


def tpr_at_fixed_fpr(fpr: torch.Tensor, tpr: torch.Tensor, targets: Sequence[float]):
    """
    Linear interpolation to get TPR at requested FPR levels.
    """
    import numpy as np

    f = fpr.cpu().numpy()
    t = tpr.cpu().numpy()
    out = {}
    for x in targets:
        # clip within [0,1], interpolate monotone curve
        x_clipped = min(max(0.0, float(x)), 1.0)
        out[str(x)] = float(np.interp(x_clipped, f, t))
    return out
