# diffusion_planner.py
# ------------------------------------------------------------
# DiffusionPlanner wrapper that plugs into environment.py's
# planner interface:
#
#   planner.plan(start, goal, bounds_lo, bounds_hi, static_obs, ...)
#       -> ref_traj: (T,3) in world coords
#
# This planner uses the trained Transformer diffusion model
# (train_diffusion_transformer.py) and the SAME static-condition
# tokenization as diffusion_dataset.py.
#
# Important design choice:
# - Diffusion was trained on subpaths of length H in *relative coords*.
# - plan() here generates a full trajectory by autoregressively
#   stitching multiple H-step subpaths:
#     p0 -> sample rel(H,3) -> update p0 = last point -> repeat
#
# Because diffusion is STATIC-only, dyn_spheres is ignored.
# ------------------------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Condition tokenization (must match diffusion_dataset.py)
# ============================================================

def _aabb_center_half(aabbs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mins = aabbs[:, 0:3]
    maxs = aabbs[:, 3:6]
    center = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5
    return center, half

def _dist_point_aabb_center_approx(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    d = np.abs(p[None, :] - center) - half
    d = np.maximum(d, 0.0)
    return np.linalg.norm(d, axis=1)

def _dist_point_sphere(p: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.maximum(np.linalg.norm(c - p[None, :], axis=1) - r.reshape(-1), 0.0)

def _scale_from_bounds(bounds_lo: np.ndarray, bounds_hi: np.ndarray) -> float:
    span = bounds_hi - bounds_lo
    s = float(np.max(span))
    return s if s > 1e-6 else 1.0

def _goal_token(goal_rel: np.ndarray, d_ctx: int) -> np.ndarray:
    # tok = np.zeros((d_ctx,), dtype=np.float32)
    # tok[0] = 1.0  # goal marker
    # tok[1:4] = goal_rel.astype(np.float32)
    tok = np.zeros(d_ctx, dtype=np.float32)
    tok[:] = goal_rel.astype(np.float32)   # (3,)
    return tok

def _aabb_token(rel_center: np.ndarray, half: np.ndarray, d_ctx: int, include_size: bool) -> np.ndarray:
    tok = np.zeros((d_ctx,), dtype=np.float32)
    tok[0] = 1.0  # AABB marker (shared slot with goal marker by design)
    tok[2:5] = rel_center.astype(np.float32)
    if include_size:
        tok[5:8] = half.astype(np.float32)
    return tok

def _sphere_token(rel_center: np.ndarray, r: float, d_ctx: int, include_size: bool) -> np.ndarray:
    tok = np.zeros((d_ctx,), dtype=np.float32)
    tok[1] = 1.0  # SPHERE marker
    tok[2:5] = rel_center.astype(np.float32)
    if include_size:
        tok[5] = float(r)
    return tok

def build_ctx_tokens(
    p0: np.ndarray,
    goal: np.ndarray,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    aabbs: np.ndarray,
    spheres: np.ndarray,
    *,
    k_obs: int,
    d_ctx: int,
    normalize: bool,
    include_size: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      ctx: (1+k_obs, d_ctx)
      ctx_mask01: (1+k_obs,) int 0/1
    """
    s = _scale_from_bounds(bounds_lo, bounds_hi) if normalize else 1.0

    g_rel = (goal - p0) / s
    gtok = _goal_token(g_rel, d_ctx)

    candidates: List[Tuple[float, str, int]] = []

    if aabbs is not None and aabbs.size > 0:
        c, half = _aabb_center_half(aabbs)
        da = _dist_point_aabb_center_approx(p0, c, half)
        for i, di in enumerate(da):
            candidates.append((float(di), "aabb", i))

    if spheres is not None and spheres.size > 0:
        c = spheres[:, 0:3]
        r = spheres[:, 3:4]
        ds = _dist_point_sphere(p0, c, r)
        for i, di in enumerate(ds):
            candidates.append((float(di), "sphere", i))

    candidates.sort(key=lambda x: x[0])
    picked = candidates[:k_obs]

    obs_toks = np.zeros((k_obs, d_ctx), dtype=np.float32)
    obs_mask = np.zeros((k_obs,), dtype=np.int64)

    if aabbs is not None and aabbs.size > 0:
        c_a, half_a = _aabb_center_half(aabbs)
    else:
        c_a, half_a = None, None

    for j, (_, typ, idx) in enumerate(picked):
        obs_mask[j] = 1
        if typ == "aabb":
            rel_c = (c_a[idx] - p0) / s
            h = (half_a[idx] / s) if normalize else half_a[idx]
            obs_toks[j] = _aabb_token(rel_c, h, d_ctx, include_size)
        else:
            c = spheres[idx, 0:3]
            r = float(spheres[idx, 3])
            rel_c = (c - p0) / s
            rr = (r / s) if normalize else r
            obs_toks[j] = _sphere_token(rel_c, rr, d_ctx, include_size)

    ctx = np.concatenate([gtok[None, :], obs_toks], axis=0)  # (1+k_obs,d_ctx)
    mask01 = np.concatenate([np.ones((1,), dtype=np.int64), obs_mask], axis=0)
    return ctx.astype(np.float32), mask01


# ============================================================
# Diffusion schedule + Transformer model (match training script)
# ============================================================

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 1e-5, 0.999)
    return betas.float()

class Diffusion:
    def __init__(self, T: int, device: torch.device):
        self.T = T
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bar[:-1]], dim=0)
        self.posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: int, ctx: torch.Tensor, ctx_mask: torch.Tensor):
        B = x.shape[0]
        tt = torch.full((B,), t, device=x.device, dtype=torch.long)
        eps_hat = model(x, tt, ctx, ctx_mask)

        beta_t = self.betas[t]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]

        mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ab * eps_hat)

        if t == 0:
            return mean
        var = self.posterior_var[t]
        noise = torch.randn_like(x)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, B: int, H: int, ctx: torch.Tensor, ctx_mask: torch.Tensor, device: torch.device):
        """DDPM sampling (slow: T steps)."""
        x = torch.randn((B, H, 3), device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, ctx, ctx_mask)
        return x

    # ============================================================
    # ✅ DDIM (step subsampling) for fast sampling
    # ============================================================

    def _make_ddim_timesteps(self, steps: int) -> List[int]:
        steps = int(max(2, steps))
        t = torch.linspace(self.T - 1, 0, steps).round().long().tolist()

        out: List[int] = []
        last = None
        for ti in t:
            if last is None or ti != last:
                out.append(int(ti))
            last = ti
        if out[-1] != 0:
            out.append(0)
        return out

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        B: int,
        H: int,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        device: torch.device,
        *,
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling:
          - steps: number of discrete sampling steps (e.g., 20~100)
          - eta: 0.0 -> deterministic DDIM (fast, stable)
        """
        x = torch.randn((B, H, 3), device=device)

        t_list = self._make_ddim_timesteps(steps)
        for i in range(len(t_list) - 1):
            t = t_list[i]
            t_prev = t_list[i + 1]

            tt = torch.full((B,), t, device=device, dtype=torch.long)
            eps = model(x, tt, ctx, ctx_mask)  # eps_theta(x_t, t)

            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[t_prev]

            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)

            # x0 prediction
            x0 = (x - sqrt_one_minus_ab_t * eps) / (sqrt_ab_t + 1e-8)
            x0 = torch.clamp(x0, -1.0, 1.0)

            # sigma for DDIM (eta=0 -> 0)
            if eta > 0.0 and t_prev > 0:
                sigma = (
                    eta
                    * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t + 1e-8))
                    * torch.sqrt(1.0 - (ab_t / (ab_prev + 1e-8)))
                )
            else:
                sigma = torch.tensor(0.0, device=device)

            c = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma * sigma, min=0.0))

            if float(sigma.item()) > 0.0:
                noise = torch.randn_like(x)
                x = torch.sqrt(ab_prev) * x0 + c * eps + sigma * noise
            else:
                x = torch.sqrt(ab_prev) * x0 + c * eps

        return x


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, cross_attn: bool):
        super().__init__()
        self.cross_attn = cross_attn
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)

        if cross_attn:
            self.xattn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
        else:
            self.xattn = None
            self.ln2 = None

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_key_padding_mask: Optional[torch.Tensor]):
        h = self.ln1(x)
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa

        if self.cross_attn:
            h = self.ln2(x)
            ca, _ = self.xattn(h, ctx, ctx, key_padding_mask=ctx_key_padding_mask, need_weights=False)
            x = x + ca

        h = self.ln3(x)
        x = x + self.ff(h)
        return x

class TrajDiffusionTransformer(nn.Module):
    def __init__(
        self,
        H: int,
        ctx_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        time_dim: int = 256,
        cross_attn_every: int = 1,
    ):
        super().__init__()
        self.H = H
        self.d_model = d_model

        self.traj_in = nn.Linear(3, d_model)
        self.ctx_in = nn.Linear(ctx_dim, d_model)

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, H, d_model))

        blocks = []
        for i in range(n_layers):
            use_cross = (cross_attn_every > 0) and ((i % cross_attn_every) == 0)
            blocks.append(TransformerBlock(d_model, n_heads, d_ff, dropout, cross_attn=use_cross))
        self.blocks = nn.ModuleList(blocks)

        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, ctx_mask01: torch.Tensor):
        B, H, _ = x_t.shape
        assert H == self.H, f"Model H={self.H}, got input H={H}"

        x = self.traj_in(x_t)
        x = x + self.pos_emb[:, :H, :]

        te = self.time_mlp(self.time_emb(t))
        x = x + te.unsqueeze(1)

        c = self.ctx_in(ctx)
        key_padding_mask = (ctx_mask01 == 0)  # True means ignore token
        for blk in self.blocks:
            x = blk(x, c, key_padding_mask)

        return self.out(x)


# ============================================================
# Static obstacles conversion from environment.py objects
# ============================================================

def _static_obs_to_arrays(static_obs: Optional[List[Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert env.static_obs (list of AABB3D / Sphere3D) to arrays:
      aabbs: (Na,6)  [xmin,ymin,zmin,xmax,ymax,zmax]
      spheres:(Ns,4) [cx,cy,cz,r]
    """
    if static_obs is None:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    aabbs = []
    spheres = []
    for o in static_obs:
        if hasattr(o, "xmin") and hasattr(o, "xmax"):
            aabbs.append([float(o.xmin), float(o.ymin), float(o.zmin), float(o.xmax), float(o.ymax), float(o.zmax)])
        elif hasattr(o, "cx") and hasattr(o, "r"):
            spheres.append([float(o.cx), float(o.cy), float(o.cz), float(o.r)])

    aabbs_arr = np.asarray(aabbs, dtype=np.float32) if len(aabbs) else np.zeros((0, 6), dtype=np.float32)
    spheres_arr = np.asarray(spheres, dtype=np.float32) if len(spheres) else np.zeros((0, 4), dtype=np.float32)
    return aabbs_arr, spheres_arr


# ============================================================
# Planner wrapper
# ============================================================

@dataclass
class DiffusionPlannerConfig:
    # generation
    stitch_steps: int = 10        # how many H-subpaths to stitch (max)
    stop_dist: float = 0.5        # stop if within this dist to goal
    max_points: int = 400         # hard cap on output points
    exec_points: int = 35         # 이동 스텝 수 (start 제외). 출력 점 개수는 exec_points+1

    # conditioning (Optional -> if None, infer from ckpt)
    k_obs: Optional[int] = None
    d_ctx: Optional[int] = None
    normalize: bool = True
    include_size: bool = True

    # If True, if stitched segment goes OOB, we clip into bounds
    clip_to_bounds: bool = False

    # ✅ add: sampler options
    sampler: str = "ddim"     # "ddpm" or "ddim"
    ddim_steps: int = 50      # 20~100 추천
    ddim_eta: float = 0.0     # 0.0이면 deterministic(빠르고 안정적)


class DiffusionPlanner:
    """
    Planner interface compatible with environment.py's UavEnvDiff:
      plan(start, goal, bounds_lo, bounds_hi, static_obs, dyn_spheres, dt, max_steps, ...) -> (T,3)
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cfg: Optional[DiffusionPlannerConfig] = None,
    ):
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path
        self.cfg = cfg if cfg is not None else DiffusionPlannerConfig()

        ckpt = torch.load(ckpt_path, map_location=self.device)
        saved_cfg = ckpt.get("cfg", {})
        saved_args = saved_cfg.get("args", {}) if isinstance(saved_cfg, dict) else {}
        saved_cond = saved_cfg.get("cond_cfg", {}) if isinstance(saved_cfg, dict) else {}

        # infer model hyperparams
        self.H = int(saved_args.get("H", 20))
        self.Tdiff = int(saved_args.get("T", 1000))

        d_model = int(saved_args.get("d_model", 256))
        n_heads = int(saved_args.get("n_heads", 8))
        n_layers = int(saved_args.get("n_layers", 6))
        d_ff = int(saved_args.get("d_ff", 1024))
        dropout = float(saved_args.get("dropout", 0.1))
        cross_every = int(saved_args.get("cross_every", 1))

        # infer conditioning unless user overrides
        if self.cfg.k_obs is None:
            self.cfg.k_obs = int(saved_cond.get("k_obs", 16))
        if self.cfg.d_ctx is None:
            self.cfg.d_ctx = int(saved_cond.get("d_ctx", 32))

        # keep normalize/include_size from saved_cond unless explicitly overridden in cfg
        self.cfg.normalize = bool(saved_cond.get("normalize", self.cfg.normalize))
        self.cfg.include_size = bool(saved_cond.get("include_size", self.cfg.include_size))

        # build model and load weights
        self.model = TrajDiffusionTransformer(
            H=self.H,
            ctx_dim=int(self.cfg.d_ctx),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            time_dim=d_model,
            cross_attn_every=cross_every,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()

        self.diffusion = Diffusion(T=self.Tdiff, device=self.device)

    # diffusion_planner.py (inside class DiffusionPlanner)

    @torch.no_grad()
    def _sample_rel(
        self,
        p0: np.ndarray,
        goal: np.ndarray,
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
        aabbs=None,
        spheres=None,
    ) -> np.ndarray:
        """
        Single-sample: returns (H,3) in relative coords (numpy float32)
        """
        p0 = np.asarray(p0, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        bounds_lo = np.asarray(bounds_lo, dtype=np.float32)
        bounds_hi = np.asarray(bounds_hi, dtype=np.float32)

        ctx_np, mask_np = build_ctx_tokens(
            p0=p0,
            goal=goal,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            aabbs=aabbs,
            spheres=spheres,
            k_obs=int(self.cfg.k_obs),
            d_ctx=int(self.cfg.d_ctx),
            normalize=bool(self.cfg.normalize),
            include_size=bool(self.cfg.include_size),
        )

        # (L,d), (L,)
        ctx = torch.from_numpy(ctx_np).to(self.device)
        mask = torch.from_numpy(mask_np).to(self.device)

        # (1,L,d), (1,L)
        ctx = ctx.unsqueeze(0)
        mask = mask.unsqueeze(0)

        if str(self.cfg.sampler).lower() == "ddim":
            rel = self.diffusion.ddim_sample(
                self.model,
                B=1,
                H=self.H,
                ctx=ctx,
                ctx_mask=mask,
                device=self.device,
                steps=int(self.cfg.ddim_steps),
                eta=float(self.cfg.ddim_eta),
            )  # (1,H,3)
        else:
            rel = self.diffusion.sample(
                self.model,
                B=1,
                H=self.H,
                ctx=ctx,
                ctx_mask=mask,
                device=self.device,
            )  # (1,H,3)

        rel = torch.clamp(rel, -5.0, 5.0)
        return rel[0].detach().cpu().numpy().astype(np.float32)


    @torch.no_grad()
    def _sample_rel_batch(
        self,
        p0: np.ndarray,          # (B,3)
        goal: np.ndarray,        # (B,3)
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
        aabbs=None,
        spheres=None,
        B: int = 8,
    ) -> np.ndarray:
        """
        Batch-sample: returns (B,H,3) in relative coords (numpy float32)
        핵심: ctx를 (B,L,d)로 repeat 해서 모델에 넣습니다.
        """
        B = int(B)
        p0 = np.asarray(p0, dtype=np.float32)      # (B,3)
        goal = np.asarray(goal, dtype=np.float32)  # (B,3)
        bounds_lo = np.asarray(bounds_lo, dtype=np.float32)
        bounds_hi = np.asarray(bounds_hi, dtype=np.float32)

        # build ctx는 "1개"만 만들고, B로 repeat (각 후보가 같은 conditioning이면 OK)
        # (지금 구조는 후보들은 diffusion 샘플링 randomness로만 갈라지는 설계)
        ctx_np, mask_np = build_ctx_tokens(
            p0=p0[0],
            goal=goal[0],
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            aabbs=aabbs,
            spheres=spheres,
            k_obs=int(self.cfg.k_obs),
            d_ctx=int(self.cfg.d_ctx),
            normalize=bool(self.cfg.normalize),
            include_size=bool(self.cfg.include_size),
        )

        ctx = torch.from_numpy(ctx_np).to(self.device)    # (L,d)
        mask = torch.from_numpy(mask_np).to(self.device)  # (L,)

        # ✅ (1,L,d)->(B,L,d), (1,L)->(B,L)
        ctx = ctx.unsqueeze(0).repeat(B, 1, 1)
        mask = mask.unsqueeze(0).repeat(B, 1)

        if str(self.cfg.sampler).lower() == "ddim":
            rel = self.diffusion.ddim_sample(
                self.model,
                B=B,
                H=self.H,
                ctx=ctx,
                ctx_mask=mask,
                device=self.device,
                steps=int(self.cfg.ddim_steps),
                eta=float(self.cfg.ddim_eta),
            )  # (B,H,3)
        else:
            rel = self.diffusion.sample(
                self.model,
                B=B,
                H=self.H,
                ctx=ctx,
                ctx_mask=mask,
                device=self.device,
            )  # (B,H,3)

        rel = torch.clamp(rel, -5.0, 5.0)
        return rel.detach().cpu().numpy().astype(np.float32)


    def plan(self, start: np.ndarray, goal: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """
        kwargs expected (from environment.py wrapper calls):
          bounds_lo, bounds_hi, static_obs, dyn_spheres(ignored), dt, max_steps ...
        """
        bounds_lo = np.asarray(kwargs.get("bounds_lo", [-50, -50, 0]), dtype=np.float32)
        bounds_hi = np.asarray(kwargs.get("bounds_hi", [50, 50, 30]), dtype=np.float32)
        static_obs = kwargs.get("static_obs", None)

        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)

        aabbs, spheres = _static_obs_to_arrays(static_obs)

        pts: List[np.ndarray] = [start.copy()]
        p0 = start.copy()

        # 원하는 "이동 스텝 수"만큼만 생성 (start 포함하면 +1)
        target_len = int(getattr(self.cfg, "exec_points", 0)) + 1
        target_len = max(2, target_len)  # 최소 2개 점

        for _ in range(int(self.cfg.stitch_steps)):
            rel = self._sample_rel(p0, goal, bounds_lo, bounds_hi, aabbs, spheres)  # (H,3)
            s = _scale_from_bounds(bounds_lo, bounds_hi) if bool(self.cfg.normalize) else 1.0
            seg = p0[None, :] + rel * s

            # remove duplicate first point if it's near p0
            if seg.shape[0] > 0 and np.linalg.norm(seg[0] - p0) < 1e-4:
                seg_use = seg[1:]
            else:
                seg_use = seg

            for p in seg_use:
                if self.cfg.clip_to_bounds:
                    p = np.minimum(np.maximum(p, bounds_lo), bounds_hi)
                pts.append(p.astype(np.float32))

                if len(pts) >= target_len:
                    break

            p0 = pts[-1].copy()

            if len(pts) >= target_len:
                break

        if len(pts) < 2:
            return None
        return np.asarray(pts, dtype=np.float32)


    def plan_candidates(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
        static_aabbs=None,
        static_spheres=None,
        num_candidates: int = 8,
    ):
        """
        Returns:
        trajs: List[np.ndarray] length=B
                each traj shape = (T,3), T = exec_points + 1 (start 포함)
        """
        B = int(num_candidates)
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        bounds_lo = np.asarray(bounds_lo, dtype=np.float32)
        bounds_hi = np.asarray(bounds_hi, dtype=np.float32)

        aabbs = static_aabbs
        spheres = static_spheres

        # ✅ "이동 스텝 수" 기준: 점 개수 = exec_points + 1 (start 포함)
        target_len = int(getattr(self.cfg, "exec_points", 50)) + 1
        target_len = max(2, target_len)

        # 후보별 현재 위치 p0 (B,3) / goal도 batch로 (B,3)
        p0 = np.repeat(start[None, :], B, axis=0)   # (B,3)
        gB = np.repeat(goal[None, :],  B, axis=0)   # (B,3)

        # 후보별 누적 점 리스트
        pts = [[p0[b].copy()] for b in range(B)]

        # scale (normalize 케이스)
        s = _scale_from_bounds(bounds_lo, bounds_hi) if bool(self.cfg.normalize) else 1.0

        for _ in range(int(self.cfg.stitch_steps)):
            # ✅ diffusion을 "한 번" 호출해서 B개 subpath를 받음
            rel = self._sample_rel_batch(
                p0=p0,
                goal=gB,
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
                aabbs=aabbs,
                spheres=spheres,
                B=B,
            )  # (B,H,3)

            # (B,H,3) absolute segment
            seg = p0[:, None, :] + rel * s

            # 각 후보별로 append
            for b in range(B):
                seg_b = seg[b]  # (H,3)

                # 중복점 제거
                if seg_b.shape[0] > 0 and np.linalg.norm(seg_b[0] - p0[b]) < 1e-4:
                    seg_use = seg_b[1:]
                else:
                    seg_use = seg_b

                for p in seg_use:
                    if self.cfg.clip_to_bounds:
                        p = np.minimum(np.maximum(p, bounds_lo), bounds_hi)
                    pts[b].append(p.astype(np.float32))
                    if len(pts[b]) >= target_len:
                        break

            # p0 업데이트
            for b in range(B):
                p0[b] = pts[b][-1].copy()

            # 모두 target_len 채웠으면 종료
            if all(len(pts[b]) >= target_len for b in range(B)):
                break

        # 길이 강제(부족하면 있는 만큼이라도 반환하지 말고, 여기서는 강제로 자름)
        trajs = []
        for b in range(B):
            arr = np.asarray(pts[b][:target_len], dtype=np.float32)
            trajs.append(arr)

        return trajs


