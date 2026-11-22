import argparse
import os

import numpy as np
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

from diffusion_planner import DiffusionPlanner, DiffusionPlannerConfig
from environment_selector import SelectorEnv


class TBMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0
        self.success_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", [])
        if dones is None:
            return True

        for i, done in enumerate(dones):
            if not done:
                continue

            self.episode_count += 1
            if bool(infos[i].get("success", False)):
                self.success_count += 1

            self.logger.record("success_rate", self.success_count / self.episode_count)

        return True


def make_env_fn(planner, seed, num_candidates, max_steps, goal_tol):
    def _init():
        env = SelectorEnv(
            planner=planner,
            num_candidates=num_candidates,
            max_steps=max_steps,
            goal_tol=goal_tol,
            #exec_horizon=exec_horizon,
        )
        env.reset(seed=seed)
        return env
    return _init



def main():
    ap = argparse.ArgumentParser()

    # diffusion
    ap.add_argument("--ckpt", type=str, default="./ckpt_step_100000.pt")
    ap.add_argument("--device", type=str, default="cuda")

    # RL
    ap.add_argument("--total_steps", type=int, default=200_000)    ## 기존 300_000
    ap.add_argument("--n_envs", type=int, default=4)   ## 기존 4
    ap.add_argument("--seed", type=int, default=0)

    # env
    ap.add_argument("--num_candidates", type=int, default=8)    ## 기존 8
    #ap.add_argument("--exec_horizon", type=int, default=30)  
    ap.add_argument("--max_steps", type=int, default=200)     ## 기존 200
    ap.add_argument("--goal_tol", type=float, default=0.5)

    # logging
    ap.add_argument("--log_dir", type=str, default="runs_selector")
    ap.add_argument("--tb_name", type=str, default="DQN_Selector")

    ap.add_argument("--subproc", action="store_true")

    args = ap.parse_args()
    set_random_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)

    # ----------------------------
    # diffusion planner
    # ----------------------------
    planner_cfg = DiffusionPlannerConfig(
        d_ctx=3,
        sampler="ddim",
        ddim_steps=50,    ## 기존 50
        ddim_eta=0.0,
        exec_points=35
    )

    planner = DiffusionPlanner(
        args.ckpt,
        device=args.device,
        cfg=planner_cfg
    )

    # ----------------------------
    # env
    # ----------------------------
    env_fns = []
    for i in range(args.n_envs):
        env_fns.append(
            make_env_fn(
                planner=planner,
                seed=args.seed + i,
                num_candidates=args.num_candidates,
                max_steps=args.max_steps,
                goal_tol=args.goal_tol,
                #exec_horizon=args.exec_horizon,
            )
        )

    VecCls = SubprocVecEnv if args.subproc else DummyVecEnv
    vec_env = VecCls(env_fns)
    vec_env = VecMonitor(vec_env)

    # ----------------------------
    # model (Discrete!)
    # ----------------------------
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,

        # 핵심 튜닝
        learning_rate=5e-5,
        buffer_size=200_000,
        learning_starts=20_000,
        batch_size=128,

        gamma=0.99,
        train_freq=8,
        target_update_interval=2_000,

        # exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.4,

        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        device=args.device,
    )

    # ----------------------------
    # train
    # ----------------------------
    callback = TBMetricsCallback()
    model.learn(
        total_timesteps=args.total_steps, 
        callback=callback, 
        tb_log_name=args.tb_name,
        progress_bar=True,)


    save_path = os.path.join(args.log_dir, "dqn_selector_6.zip")
    model.save(save_path)
    print(f"[DONE] saved to {save_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
