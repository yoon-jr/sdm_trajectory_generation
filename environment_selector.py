import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, List


def l2(a, b):
    return float(np.linalg.norm(a - b))


def _path_length(traj: np.ndarray) -> float:
    if traj.shape[0] < 2:
        return 0.0
    dif = traj[1:] - traj[:-1]
    return float(np.linalg.norm(dif, axis=1).sum())

def _smoothness_cost(traj: np.ndarray) -> float:
    """
    Smoothness proxy using 2nd finite difference:
      a_i = p_{i+1} - 2 p_i + p_{i-1}
    Return mean ||a_i|| (scale: coord unit per step^2).
    """
    if traj.shape[0] < 3:
        return 0.0
    a = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]   # (T-2, 3)
    return float(np.linalg.norm(a, axis=1).mean())

class SelectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        planner,
        num_candidates: int = 4,
        max_steps: int = 100,
        goal_tol: float = 2.0,
        #exec_horizon: int = 30,
        cand_feat_dim: int = 3,

        # ✅ reward weights (추가)
        w_dist: float = 0.01,
        w_progress: float = 1.0,
        w_success: float = 100.0,
        w_smooth: float = 0.3,     # smoothness penalty weight
        w_step: float = 0.005,       # per-decision(step) penalty
    ):
        super().__init__()

        self.planner = planner
        self.num_candidates = int(num_candidates)
        self.max_steps = int(max_steps)
        self.goal_tol = float(goal_tol)
        #self.exec_horizon = int(exec_horizon)

        # action = candidate trajectory index (매 step마다 선택)
        self.action_space = gym.spaces.Discrete(self.num_candidates)

        # 후보 요약 feature (궤적 전체 좌표/노이즈/타임스텝 등 diffusion 내부 정보는 포함하지 않음)
        # feat = [final_dist_to_goal, path_length, straightness_ratio] (3)
        self.cand_feat_dim = int(cand_feat_dim)
        obs_dim = 3 + 3 + (self.num_candidates * self.cand_feat_dim)  # pos + goal_rel + K*feat

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # runtime
        self.pos = None
        self.goal = None
        self.step_count = 0

        self.candidate_trajs: List[np.ndarray] = []
        self.candidate_feats: np.ndarray = np.zeros((self.num_candidates, self.cand_feat_dim), dtype=np.float32)

        # ✅ store weights
        self.w_dist = float(w_dist)
        self.w_progress = float(w_progress)
        self.w_success = float(w_success)
        self.w_smooth = float(w_smooth)
        self.w_step = float(w_step)


    # --------------------------------------------------
    # reset
    # --------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        self.step_count = 0

        # start / goal
        if options and "start" in options:
            self.pos = np.asarray(options["start"], dtype=np.float32)
        else:
            self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if options and "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float32)
        else:
            self.goal = np.array([12.5, 12.5, 12.5], dtype=np.float32)

        # reset에서 "현재 pos -> goal" 후보 생성
        self._regen_candidates()

        return self._make_obs(), {}

    # --------------------------------------------------
    # candidate generation + feature summary
    # --------------------------------------------------
    def _summarize_traj(self, traj: np.ndarray) -> np.ndarray:
        # traj: (T,3), traj[0] ~= current pos, traj[-1] near goal (ideally)
        final_dist = l2(traj[-1], self.goal)
        plen = _path_length(traj)
        straight = l2(traj[0], traj[-1])
        straightness_ratio = float(straight / (plen + 1e-6))  # 0~1 근처

        return np.array([final_dist, plen, straightness_ratio], dtype=np.float32)

    def _regen_candidates(self):
        self.candidate_trajs = []
        feats = []

        bounds_lo = np.asarray([0, 0, 0], dtype=np.float32)
        bounds_hi = np.asarray([15, 15, 15], dtype=np.float32)

        '''
        # 단일 궤적 생성
        # planner가 실패할 수도 있으니, 일정 횟수까지 시도
        max_tries = self.num_candidates * 5
        tries = 0
        while len(self.candidate_trajs) < self.num_candidates and tries < max_tries:
            tries += 1
            traj = self.planner.plan(start=self.pos, goal=self.goal)
            if traj is None:
                continue

            traj = np.asarray(traj, dtype=np.float32)
            if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 2:
                continue

            # 시작점이 현재 pos와 너무 다르면, 첫 점을 강제로 현재 pos로 맞춤 (안정성)
            if l2(traj[0], self.pos) > 1e-3:
                traj = traj.copy()
                traj[0] = self.pos.copy()

            self.candidate_trajs.append(traj)
            feats.append(self._summarize_traj(traj))
        '''

        # 한 번에 B개 후보 생성 (Diffusion 내부 B=K)
        trajs = self.planner.plan_candidates(
            start=self.pos,
            goal=self.goal,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            static_aabbs=getattr(self, "static_aabbs", None),
            static_spheres=getattr(self, "static_spheres", None),
            num_candidates=self.num_candidates,
        )

        for traj in trajs:
            traj = np.asarray(traj, dtype=np.float32)
            if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 2:
                continue

            if l2(traj[0], self.pos) > 1e-3:
                traj = traj.copy()
                traj[0] = self.pos.copy()

            self.candidate_trajs.append(traj)
            feats.append(self._summarize_traj(traj))

        if len(self.candidate_trajs) < self.num_candidates:
            raise RuntimeError(
                f"DiffusionPlanner failed: generated {len(self.candidate_trajs)}/{self.num_candidates} candidates."
            )

        self.candidate_feats = np.stack(feats, axis=0).astype(np.float32)  # (K,feat_dim)

    # --------------------------------------------------
    # step
    # --------------------------------------------------
    def step(self, action: int):
        prev_pos = self.pos.copy()
        prev_dist = l2(prev_pos, self.goal)

        # action -> candidate 선택
        a = int(np.clip(action, 0, self.num_candidates - 1))
        traj = self.candidate_trajs[a]

        # 선택된 trajectory를 끝까지 실행
        exec_idx = traj.shape[0] - 1
        self.pos = traj[exec_idx].copy()

        self.step_count += 1

        # termination (기존과 동일한 의미로 유지: success/timeout)
        dist_goal = l2(self.pos, self.goal)
        success = dist_goal <= self.goal_tol
        timeout = self.step_count >= self.max_steps

        terminated = bool(success)
        truncated = bool(timeout and not terminated)

        # reward (실제 실행 결과 기반: pos 업데이트 후 dist 변화를 사용)
        # 거리 기반 패널티
        distance_penalty = -self.w_dist * dist_goal
        # 전진에 대한 보상
        progress_reward = self.w_progress * (prev_dist - dist_goal)
        # 선택한 traj의 굴곡/급변 억제
        smooth_cost = _smoothness_cost(traj)
        smooth_penalty = -self.w_smooth * smooth_cost
        # 도달까지 걸린 step 수에 비례한 패널티
        step_penalty = -self.w_step
        # 성공 보상
        success_reward = self.w_success if success else 0.0

        reward = distance_penalty + progress_reward + smooth_penalty + step_penalty + success_reward

        # --- DEBUG: candidate quality / scale check ---
        if self.step_count % 1000 == 0:
            cand_lens = [int(t.shape[0]) for t in self.candidate_trajs]

            # 끝점(= 실행 결과) 기준 거리만 보자
            final_dists = [float(np.linalg.norm(t[-1] - self.goal)) for t in self.candidate_trajs]

            # 후보 전체 좌표 범위(스케일 감)
            all_pts = np.concatenate(self.candidate_trajs, axis=0)
            xyz_min = all_pts.min(axis=0)
            xyz_max = all_pts.max(axis=0)

            print(
                f"[cand-check] env_step={self.step_count} "
                f"pos={self.pos.round(3)} goal={self.goal.round(3)} "
                f"lens={cand_lens} "
                f"min_final_dist={min(final_dists):.3f} "
                f"final_dists={np.round(final_dists,3)} "
                f"traj_xyz_min={np.round(xyz_min,3)} traj_xyz_max={np.round(xyz_max,3)}"
            )
        # --- DEBUG end ---

        info = {
            "chosen_traj_id": a,
            "exec_idx": exec_idx,
            "traj_len": int(traj.shape[0]),
            "dist_goal": float(dist_goal),
            "success": bool(success),
            "r_progress": float(progress_reward),
            "r_dist": float(distance_penalty),
            "r_success": float(success_reward),
            "r_smooth": float(smooth_penalty),
            "r_step": float(step_penalty),
        }

        # 다음 step을 위한 후보 재생성 (종료면 굳이 안 만들어도 됨)
        if not (terminated or truncated):
            self._regen_candidates()

        if terminated or truncated:
            info["episode"] = {
                "r": float(reward),
                "l": int(self.step_count),
                "dist_goal_final": float(dist_goal),
                "last_chosen_traj_id": int(a),
            }

        return self._make_obs(), float(reward), terminated, truncated, info

    # --------------------------------------------------
    # observation
    # --------------------------------------------------
    def _make_obs(self):
        goal_rel = (self.goal - self.pos).astype(np.float32)  # (3,)
        feats_flat = self.candidate_feats.reshape(-1).astype(np.float32)  # (K*feat_dim,)

        obs = np.concatenate([self.pos, goal_rel, feats_flat], axis=0).astype(np.float32)
        return obs
