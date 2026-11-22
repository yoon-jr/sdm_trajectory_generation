# inference_rl.py
# python inference_rl_final.py --goal_id 1 --round_trip 1

import argparse
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from diffusion_planner import DiffusionPlanner, DiffusionPlannerConfig
from environment_selector import SelectorEnv


# -----------------------------
# utils
# -----------------------------
def l2(a, b):
    return float(np.linalg.norm(a - b))


def save_rollout_csv(path: Path, rows: list):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def resample_traj_by_arclength(traj: np.ndarray, n_points: int = 10) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.shape[0] <= 1:
        return traj

    dif = traj[1:] - traj[:-1]
    seg = np.linalg.norm(dif, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    if total < 1e-8:
        return np.repeat(traj[:1], n_points, axis=0)

    s_new = np.linspace(0.0, total, n_points)
    out = np.zeros((n_points, 3), dtype=np.float32)
    for d in range(3):
        out[:, d] = np.interp(s_new, s, traj[:, d])
    return out


def plot_top_view_roundtrip(
    leg1_xyz: np.ndarray,
    leg2_xyz: np.ndarray | None,
    start: np.ndarray,
    goal: np.ndarray,
    out_path: Path,
):
    plt.figure()

    # leg1: start -> goal
    plt.plot(leg1_xyz[:, 0], leg1_xyz[:, 1], marker="o", markersize=3, alpha=0.5, label="leg1 (go)")

    # leg2: goal -> start (있을 때만)
    if leg2_xyz is not None and len(leg2_xyz) > 0:
        plt.plot(leg2_xyz[:, 0], leg2_xyz[:, 1], marker="o", markersize=3, alpha=0.5, label="leg2 (return)")

    plt.scatter([start[0]], [start[1]], s=50, c="red", marker="s", label="start", zorder=10)
    plt.scatter([goal[0]], [goal[1]], marker="*", s=120, c="red", label="goal", zorder=10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Top view (x-y)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_side_view_roundtrip(
    leg1_xyz: np.ndarray,
    leg2_xyz: np.ndarray | None,
    start: np.ndarray,
    goal: np.ndarray,
    out_path: Path,
):
    plt.figure()

    # leg1: start -> goal
    plt.plot(leg1_xyz[:, 0], leg1_xyz[:, 2], marker="o", markersize=3, alpha=0.5, label="leg1 (go)")

    # leg2: goal -> start
    if leg2_xyz is not None and len(leg2_xyz) > 0:
        plt.plot(leg2_xyz[:, 0], leg2_xyz[:, 2], marker="o", markersize=3, alpha=0.5, label="leg2 (return)")

    plt.scatter([start[0]], [start[2]], marker="s", s=50, c="red", label="start", zorder=10)
    plt.scatter([goal[0]], [goal[2]], marker="*", s=120, label="goal", c="red", zorder=10)

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Side view (x-z)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_3d_roundtrip(
    leg1_xyz: np.ndarray,
    leg2_xyz: np.ndarray | None,
    start: np.ndarray,
    goal: np.ndarray,
    out_path: Path,
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # leg1
    ax.plot(leg1_xyz[:, 0], leg1_xyz[:, 1], leg1_xyz[:, 2], marker="o", markersize=3, alpha=0.5, label="leg1 (go)")

    # leg2
    if leg2_xyz is not None and len(leg2_xyz) > 0:
        ax.plot(leg2_xyz[:, 0], leg2_xyz[:, 1], leg2_xyz[:, 2], marker="o", markersize=3, alpha=0.5, label="leg2 (return)")

    ax.scatter(start[0], start[1], start[2], marker="s", s=50, c="red", label="start")
    ax.scatter(goal[0], goal[1], goal[2], marker="*", s=120, c="red", label="goal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trajectory 3D")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_3d_fix(
    leg1_xyz: np.ndarray,
    leg2_xyz: np.ndarray | None,
    start: np.ndarray,
    goal: np.ndarray,
    out_path: Path,
    lim_lo: float = 0.0,
    lim_hi: float = 15.0,
):
    """
    3D plot 저장 (축 범위 고정: x,y,z 모두 [lim_lo, lim_hi])
    - 왕복이면 leg1/leg2를 서로 다른 색으로 자동 표시(matplotlib 기본 컬러 사이클)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # leg1: start -> goal
    ax.plot(
        leg1_xyz[:, 0], leg1_xyz[:, 1], leg1_xyz[:, 2],
        marker="o", markersize=3, alpha=0.5, label="leg1 (go)"
    )

    # leg2: goal -> start (있을 때만)
    if leg2_xyz is not None and len(leg2_xyz) > 0:
        ax.plot(
            leg2_xyz[:, 0], leg2_xyz[:, 1], leg2_xyz[:, 2],
            marker="o", markersize=3, alpha=0.5, label="leg2 (return)"
        )

    # start / goal
    ax.scatter(start[0], start[1], start[2], marker="s", s=50, c="red", label="start")
    ax.scatter(goal[0], goal[1], goal[2], marker="*", s=120, c="red", label="goal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trajectory 3D (fixed scale)")

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_zlim(lim_lo, lim_hi)

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



# -----------------------------
# goal mapping (1..27)
# 3x3x3 grid:
#   x in [10,20,30], y in [0,10,20], z in [0,10,20]
# order: x fastest, then y, then z
# so id=1 -> (10,0,0), id=2 -> (20,0,0), ...
# -----------------------------
def goal_from_id(goal_id: int) -> np.ndarray:
    if not (1 <= goal_id <= 27):
        raise ValueError("goal_id must be in [1, 27].")

    goals = np.array([
        [2.5,  2.5,  2.5],   [2.5,  2.5,  7.5],   [2.5,  2.5, 12.5],
        [7.5,  2.5,  2.5],   [7.5,  2.5,  7.5],   [7.5,  2.5, 12.5],
        [12.5, 2.5,  2.5],   [12.5, 2.5,  7.5],   [12.5, 2.5, 12.5],

        [2.5,  7.5,  2.5],   [2.5,  7.5,  7.5],   [2.5,  7.5, 12.5],
        [7.5,  7.5,  2.5],   [7.5,  7.5,  7.5],   [7.5,  7.5, 12.5],
        [12.5, 7.5,  2.5],   [12.5, 7.5,  7.5],   [12.5, 7.5, 12.5],

        [2.5, 12.5,  2.5],   [2.5, 12.5,  7.5],   [2.5, 12.5, 12.5],
        [7.5, 12.5,  2.5],   [7.5, 12.5,  7.5],   [7.5, 12.5, 12.5],
        [12.5,12.5,  2.5],   [12.5,12.5,  7.5],   [12.5,12.5, 12.5],
    ], dtype=np.float32)


    return goals[goal_id - 1].copy()



# -----------------------------
# run one leg: (start -> goal)
# -----------------------------
def run_leg(
    *,
    env: SelectorEnv,
    model: DQN,
    ep: int,
    leg: int,
    start: np.ndarray,
    goal: np.ndarray,
    deterministic: bool,
    viz_points: int,
    trip_mode: str,
    weight_ref: dict,  # {"value": float}
    global_point_i_ref: dict,  # {"value": int}
    seed: int,
):
    obs, _ = env.reset(seed=seed, options={"start": start, "goal": goal})

    rollout_xyz = [env.pos.copy()]
    rollout_rows = []

    # reset 직후 기록 (step=0)
    rollout_rows.append({
        "episode": ep,
        "leg": leg,
        "trip_mode": trip_mode,                 # "one_way" / "round_trip"
        "weight": float(weight_ref["value"]),
        "start_x": float(start[0]),
        "start_y": float(start[1]),
        "start_z": float(start[2]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
        "step": 0,
        "row_type": "step_end",
        "traj_point_i": -1,
        "global_point_i": int(global_point_i_ref["value"]),
        "x": float(env.pos[0]),
        "y": float(env.pos[1]),
        "z": float(env.pos[2]),
        "dist_goal": float(l2(env.pos, env.goal)),
        "reward": 0.0,
        "chosen_traj_id": -1,
        "success": False,
    })
    global_point_i_ref["value"] += 1

    terminated = False
    truncated = False
    step_i = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=bool(deterministic))
        a = int(action)

        # step 호출 전에 chosen candidate traj 확보
        chosen_traj = np.asarray(env.candidate_trajs[a], dtype=np.float32)

        # 시각화용 리샘플 (dense 기록)
        viz_traj = resample_traj_by_arclength(chosen_traj, n_points=int(viz_points))

        for j, p in enumerate(viz_traj[1:], start=1):
            rollout_xyz.append(p.copy())
            rollout_rows.append({
                "episode": ep,
                "leg": leg,
                "trip_mode": trip_mode,
                "weight": float(weight_ref["value"]),
                "start_x": float(start[0]),
                "start_y": float(start[1]),
                "start_z": float(start[2]),
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "goal_z": float(goal[2]),
                "step": step_i + 1,
                "row_type": "dense",
                "traj_point_i": int(j),
                "global_point_i": int(global_point_i_ref["value"]),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "dist_goal": float(l2(p, env.goal)),
                "reward": np.nan,
                "chosen_traj_id": int(a),
                "success": False,
            })
            global_point_i_ref["value"] += 1

        # 실제 env.step 실행
        obs, reward, terminated, truncated, info = env.step(a)
        step_i += 1

        rollout_xyz.append(env.pos.copy())
        dist_goal = l2(env.pos, env.goal)
        success_now = bool(info.get("success", False))

        rollout_rows.append({
            "episode": ep,
            "leg": leg,
            "trip_mode": trip_mode,
            "weight": float(weight_ref["value"]),
            "start_x": float(start[0]),
            "start_y": float(start[1]),
            "start_z": float(start[2]),
            "goal_x": float(goal[0]),
            "goal_y": float(goal[1]),
            "goal_z": float(goal[2]),
            "step": step_i,
            "row_type": "step_end",
            "traj_point_i": -1,
            "global_point_i": int(global_point_i_ref["value"]),
            "x": float(env.pos[0]),
            "y": float(env.pos[1]),
            "z": float(env.pos[2]),
            "dist_goal": float(dist_goal),
            "reward": float(reward),
            "chosen_traj_id": int(info.get("chosen_traj_id", a)),
            "success": bool(success_now),
        })
        global_point_i_ref["value"] += 1

    traj_xyz = np.asarray(rollout_xyz, dtype=np.float32)
    final_dist = l2(traj_xyz[-1], env.goal)
    success = final_dist <= float(env.goal_tol)

    return {
        "rows": rollout_rows,
        "xyz": traj_xyz,
        "success": bool(success),
        "steps": int(step_i),
        "final_dist": float(final_dist),
    }


def main():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--ckpt", type=str, default="./ckpt_step_100000.pt")
    ap.add_argument("--model", type=str, default="./runs_selector/dqn_selector_05.zip")
    ap.add_argument("--out_dir", type=str, default="runs_infer_ch1")

    # run config
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true", dest="deterministic")
    ap.set_defaults(deterministic=True)

    # env / planner config
    ap.add_argument("--num_candidates", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--goal_tol", type=float, default=0.5)

    # planner sampling
    ap.add_argument("--exec_points", type=int, default=35)
    ap.add_argument("--ddim_steps", type=int, default=50)

    # viz
    ap.add_argument("--viz_points", type=int, default=3)

    # NEW: goal + trip mode
    ap.add_argument("--goal_id", type=int, required=True, help="1..27")
    ap.add_argument("--round_trip", type=int, default=0, help="1=round trip, 0=one way")

    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # fixed start
    fixed_start = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # goal from id
    goal = goal_from_id(int(args.goal_id))

    trip_mode = "round_trip" if int(args.round_trip) == 1 else "one_way"

    # diffusion planner
    planner_cfg = DiffusionPlannerConfig(
        d_ctx=3,
        sampler="ddim",
        ddim_steps=int(args.ddim_steps),
        ddim_eta=0.0,
        exec_points=int(args.exec_points),
    )
    planner = DiffusionPlanner(args.ckpt, device=args.device, cfg=planner_cfg)

    # env
    env = SelectorEnv(
        planner=planner,
        num_candidates=int(args.num_candidates),
        max_steps=int(args.max_steps),
        goal_tol=float(args.goal_tol),
    )

    # load RL model
    model = DQN.load(args.model, device=args.device)

    # episodes
    summary_rows = []

    for ep in range(1, int(args.episodes) + 1):
        ep_dir = out_root / f"ep_{ep:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        weight_ref = {"value": 0.0}
        global_point_i_ref = {"value": 0}

        all_rows = []
        all_xyz_list = []

        # -----------------------
        # LEG 1: start -> goal
        # -----------------------
        leg1 = run_leg(
            env=env,
            model=model,
            ep=ep,
            leg=1,
            start=fixed_start,
            goal=goal,
            deterministic=bool(args.deterministic),
            viz_points=int(args.viz_points),
            trip_mode=trip_mode,
            weight_ref=weight_ref,
            global_point_i_ref=global_point_i_ref,
            seed=int(args.seed + ep * 10 + 1),
        )

        all_rows.extend(leg1["rows"])
        all_xyz_list.append(leg1["xyz"])

        # weight update rules
        if leg1["success"]:
            weight_ref["value"] += 25.0

        # -----------------------
        # LEG 2 (only if round_trip AND leg1 success):
        # goal -> start
        # -----------------------
        leg2_success = False
        leg2_steps = 0
        leg2_final_dist = np.nan

        if int(args.round_trip) == 1 and leg1["success"]:
            leg2 = run_leg(
                env=env,
                model=model,
                ep=ep,
                leg=2,
                start=goal,
                goal=fixed_start,
                deterministic=bool(args.deterministic),
                viz_points=int(args.viz_points),
                trip_mode=trip_mode,
                weight_ref=weight_ref,
                global_point_i_ref=global_point_i_ref,
                seed=int(args.seed + ep * 10 + 2),
            )
            all_rows.extend(leg2["rows"])
            all_xyz_list.append(leg2["xyz"])

            leg2_success = bool(leg2["success"])
            leg2_steps = int(leg2["steps"])
            leg2_final_dist = float(leg2["final_dist"])

            if leg2_success:
                weight_ref["value"] -= 25.0  # return to 0

        # concat xyz for plots
        leg1_xyz = leg1["xyz"]
        leg2_xyz = None

        if int(args.round_trip) == 1 and leg1["success"]:
            leg2_xyz = leg2["xyz"]

        # save per-episode rollout
        save_rollout_csv(ep_dir / "inference_traj.csv", all_rows)

        # plot (왕복이면 두 색, 단방향이면 leg1만)
        plot_top_view_roundtrip(leg1_xyz, leg2_xyz, fixed_start, goal, ep_dir / "top_view.png")
        plot_side_view_roundtrip(leg1_xyz, leg2_xyz, fixed_start, goal, ep_dir / "side_view.png")
        plot_3d_roundtrip(leg1_xyz, leg2_xyz, fixed_start, goal, ep_dir / "traj_3d.png")
        plot_3d_fix(leg1_xyz, leg2_xyz, fixed_start, goal, ep_dir / "traj_3d_fix.png")

        
        # summary row (weight 포함)
        summary_rows.append({
            "episode_id": ep,
            "seed_base": int(args.seed),
            "goal_id": int(args.goal_id),
            "goal_x": float(goal[0]),
            "goal_y": float(goal[1]),
            "goal_z": float(goal[2]),
            "trip_mode": trip_mode,
            "round_trip": int(args.round_trip),
            "weight": float(weight_ref["value"]),
            "leg1_success": bool(leg1["success"]),
            "leg1_final_dist": float(leg1["final_dist"]),
            "leg1_steps": int(leg1["steps"]),
            "leg2_ran": bool(int(args.round_trip) == 1 and leg1["success"]),
            "leg2_success": bool(leg2_success),
            "leg2_final_dist": float(leg2_final_dist),
            "leg2_steps": int(leg2_steps),
        })

    # overall summary.csv
    save_rollout_csv(out_root / "summary.csv", summary_rows)

    print(f"[DONE] saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
