import torch
import numpy as np
import os
import argparse
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../locomotion')))
from rsl_rl.runners import OnPolicyRunner
from wheel_legged_env import WheelLeggedEnv
import copy
import genesis as gs


def generate_policy_pt():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="wheel-legged-walking-v25000")
    parser.add_argument("--ckpt", type=int, default=25000)
    agrs = parser.parse_args()

    gs.init(backend=gs.cuda, logging_level="warning")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../locomotion'))
    log_dir = os.path.join(base_dir, "logs", agrs.exp_name)
    print(f"log_dir: {log_dir}")
    print(f'cfgs path: {os.path.join(log_dir, "cfgs.pkl")}')
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
    terrain_cfg["terrain"] = True
    terrain_cfg["eval"] = "agent_eval_gym" 

    env = WheelLeggedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg,
        terrain_cfg=terrain_cfg,
        robot_morphs="urdf",
        show_viewer=False,
        train_mode=False
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{agrs.ckpt}.pt")
    runner.load(resume_path)
    model = copy.deepcopy(runner.alg.actor_critic.actor).to("cpu")
    torch.jit.script(model).save(log_dir + "/policy.pt")

    print(f"load the model using jit, save to {log_dir}/policy.pt")

if __name__ == "__main__":
    generate_policy_pt()


