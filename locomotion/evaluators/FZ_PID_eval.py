import argparse
import os
import pickle
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from locomotion.environments.FZ_PID_env_c import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import sys

from utils import gamepad
import copy
from datetime import datetime
# 路径处理

# [新增] 策略导出封装类：将 归一化层 和 Actor网络 融合
# 请将此代码块粘贴在 import 之后，def main(): 之前
class PolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        # 1. 深度拷贝训练好的归一化层 (包含均值和方差)
        self.obs_normalizer = copy.deepcopy(actor_critic.obs_normalizer)
        # 2. 深度拷贝训练好的 Actor 网络
        self.actor = copy.deepcopy(actor_critic.actor)
        
        # 3. 设置为评估模式 (关键！这会冻结归一化参数，防止测试时被污染)
        self.eval()

    def forward(self, obs):
        # [核心数据流] 原始观测 -> 自动归一化 -> 神经网络 -> 动作
        norm_obs = self.obs_normalizer(obs)
        action = self.actor(norm_obs)
        return action

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking-v15000")
    parser.add_argument("--ckpt", type=int, default=15000)

    args = parser.parse_args()


# ================= 1. 路径管理 (这是优雅的关键) =================
    # 源文件路径 (读取配置和模型)
    from torch.utils.tensorboard import SummaryWriter
    # 训练日志目录：locomotion/logs/[实验名]
    train_log_dir = os.path.join(parent_dir, "logs", args.exp_name)
    model_path = os.path.join(train_log_dir, f"model_{args.ckpt}.pt")
    cfg_path = os.path.join(train_log_dir, "cfgs.pkl")

    # # 初始化日志（记录对比数据）
    # log_dir = f"./control_debug_{args.control_mode}"
    # os.makedirs(log_dir, exist_ok=True)


    # 目标日志路径 (写入本次评估数据)
    # 结构: logs/{exp_name}_eval/{日期_时间}_ckpt{ckpt}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_log_dir = os.path.join(current_dir, "logs", f"{args.exp_name}_eval", f"{timestamp}_ckpt{args.ckpt}")
    os.makedirs(eval_log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 启动评估模式")
    print(f"📂 读取模型: {model_path}")
    print(f"📝 日志存放: {eval_log_dir}  <-- 独立的评估日志！")
    print(f"{'='*60}\n")



    gs.init(backend=gs.cuda,logging_level="warning")
    gs.device="cuda:0"
# 读取训练时的配置
    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(f)    
        
    # env_cfg["simulate_action_latency"] = False
    # terrain_cfg["terrain"] = True
    # terrain_cfg["eval"] = "slope_10deg_fit"
    # terrain_cfg["horizontal_scale"] = 0.05
    # terrain_cfg["vertical_scale"] = 0.004

    # terrain_cfg["eval"] = "wave"
    # terrain_cfg["horizontal_scale"] = 0.1
    # terrain_cfg["vertical_scale"] = 0.001
    
    # terrain_cfg["eval"] = "slope_15deg_fit"
    # terrain_cfg["horizontal_scale"] = 0.05
    # terrain_cfg["vertical_scale"] = 0.004

    # terrain_cfg["eval"] = "slope_20deg_fit"
    # terrain_cfg["horizontal_scale"] = 0.05
    # terrain_cfg["vertical_scale"] = 0.005  

    # env_cfg["kp"] = 40
    # env_cfg["wheel_action_scale"] = 5
    # env_cfg["joint_damping"] = 0
    # env_cfg["wheel_damping"] = 0
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
        show_viewer=True,
        train_mode=False,
    )
# ================= 3. [核心] 注入独立的 Logger =================
    # 这里我们创建一个全新的 Writer，指向 eval_log_dir
    # 这样产生的数据绝对不会污染训练日志
    env.set_logger(SummaryWriter(log_dir=eval_log_dir))

    
    print(reward_cfg)
    
# 1. 初始化 Runner
    runner = OnPolicyRunner(env, train_cfg, eval_log_dir, device="cuda:0")
    runner.load(model_path)
    
    print(f"\n📦 正在处理模型: {model_path}")

    # ================= [核心修改] =================
    # A. 封装导出 (是为了存文件)
    # 先把模型搬到 CPU，做成通用的 JIT 文件
    policy_exporter = PolicyExporter(runner.alg.actor_critic).to("cpu") 
    scripted_policy = torch.jit.script(policy_exporter)
    
    # 生成清晰的导出文件名：实验名_迭代次数_fused.pt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_basename = f"{args.exp_name}_ckpt{args.ckpt}_fused_{timestamp}.pt"
    export_path = os.path.join(eval_log_dir, export_basename)
    scripted_policy.save(export_path)
    print(f"✅ 策略已导出(CPU格式): {export_path}")
    
    # 同时保存一个简化的链接文件，方便快速使用
    simple_path = os.path.join(eval_log_dir, "policy_fused.pt")
    if os.path.exists(simple_path):
        os.remove(simple_path)
    os.symlink(export_basename, simple_path)
    print(f"🔗 创建符号链接: policy_fused.pt -> {export_basename}")

    # B. 加载测试 (是为了跑仿真)
    # 这里完全回复你之前的习惯：从磁盘加载 -> 设为eval -> 搬到GPU
    print("\n--- 模型加载测试 ---")
    try:
        # 1. 从磁盘加载刚才保存的文件
        loaded_policy = torch.jit.load(export_path)
        
        # 2. [关键] 搬运到 GPU (这样跑得快！)
        loaded_policy.to('cuda:0')
        
        # 3. [关键] 再次确认 eval 模式 (虽然 Exporter 里写了，但再写一次双重保险)
        loaded_policy.eval()
        
        print(f"模型加载成功! 当前设备: {next(loaded_policy.parameters()).device}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()
    # ================= [修改结束] =================

    step_count = 0
    obs, _ = env.reset()   # 重置环境，获取初始观测值
    pad = gamepad.control_gamepad(command_cfg,[0.1,1.0,1.57,0.0125])  # 初始化控制设备（手柄/键盘）
    with torch.no_grad():
        
        while True:
        # for i in range(10):
            # actions = policy(obs)
            actions = loaded_policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            comands,reset_flag = pad.get_commands()   # 读取手柄/键盘输入
            env.set_commands(0,comands)
            if step_count % 10 == 0:
                print(f"pitch: {env.lqr_controller.pitch}, yaw: {env.lqr_controller.yaw}, roll: {env.lqr_controller.roll}")
                print(f"pitch_dot: {env.lqr_controller.pitch_dot}, velocity: {env.lqr_controller.robot_x_velocity}")
                print(f"vel_error: {env.lqr_controller.velocity_d - env.lqr_controller.robot_x_velocity}")
                print(f"pitch_offset: {env.lqr_controller.pitch_offset}")
                print(f"command: {env.commands}")
                print(f"acc_x: {env.base_lin_acc}")
                print(f"projected_gravity: {env.projected_gravity}")
                # print(f"step_time: {step_time:.4f} sec")
                print("====================================================================")
            step_count += 1
            if reset_flag:
                env.reset()


if __name__ == "__main__":
    main()

