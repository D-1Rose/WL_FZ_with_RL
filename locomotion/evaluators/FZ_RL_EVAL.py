import argparse
import os
import pickle

import torch
from environments.FZ_RL_ENV import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
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
    train_log_dir = os.path.join(current_dir, "logs", args.exp_name)
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

    terrain_cfg["eval"] = "wave"
    terrain_cfg["horizontal_scale"] = 0.1
    terrain_cfg["vertical_scale"] = 0.001
    
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
    
    export_path = os.path.join(eval_log_dir, "policy_fused.pt")
    scripted_policy.save(export_path)
    print(f"✅ 策略已导出(CPU格式): {export_path}")

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


    obs, _ = env.reset()   # 重置环境，获取初始观测值

    # pad = gamepad.control_gamepad(command_cfg,[1.0,1.0,1.0,0.0125])  # 初始化控制设备（手柄/键盘）
# [修改2] 定义自动化指令序列
    # 格式: (开始时间s, 结束时间s, [vel_x, vel_y, ang_vel_yaw, height_target])
    # 假设向前速度为 0.4 m/s，高度保持 0.28m
    vel = -2.0

    test_schedule = [
        (0.0,   0.1,  [-0.0,  0.0, 0.0, 0.28]),   # 0-10s:  向前走
        (0.1,   0.5,  [-0.8,  0.0, 0.0, 0.28]),   # 0-10s:  向前走
        (0.5,   0.8,  [-1.5,  0.0, 0.0, 0.28]),   # 0-10s:  向前走
        (0.8,  10.0,  [vel,  0.0, 0.0, 0.28]),   # 2-10s: 向前走
        (10.0,  20.0,  [vel,  0.0, 0.0, 0.28]),   # 10-20s: 停住
        # (20.0,  30.0,  [vel, 0.0, 0.0, 0.28]),   # 20-30s: 向后走
        # (30.0,  50.0,  [vel,  0.0, 0.0, 0.28]),   # 30-40s: 停住 (结束)
    ]
    
    # 设定总时长
    max_test_time = 9.0
    print(f"\n🚀 启动自动化测试，总时长: {max_test_time}s")
    
    # 初始化时间计数器
    sim_dt = env.dt          # 获取仿真步长 (通常 0.01s)

    
    # [修改 1] 获取准确的 RL 控制步长 (通常 10 * 0.002 = 0.02s)
    # 如果 env 内部没有直接暴露 decimation，可通过 env_cfg 获取
    control_decimation = env_cfg.get("control_decimation", 10) 
    rl_dt = sim_dt * control_decimation  
    
    current_sim_time = 0.0   # 当前仿真时间
    current_step = 0         # 当前步数

    try:
        with torch.no_grad():
            while current_sim_time < max_test_time:
                # [修改 2] 记录当前控制步的挂钟起始时间
                loop_start = time.perf_counter()
                
                # 1. 策略推理
                actions = loaded_policy(obs)
                
                # 2. 环境步进 (这里内部已经跑了 decimation 次物理计算)
                obs, _, rews, dones, infos = env.step(actions)

                # 3. 查找指令
                current_command = [0.0, 0.0, 0.0, 0.28] # 默认指令
                for (start, end, cmd) in test_schedule:
                    if start <= current_sim_time < end:
                        current_command = cmd
                        break 
                
                # 4. 发送指令
                env.set_commands(0, current_command)

                # 5. [修改 3] 更新仿真时间，必须加上 rl_dt (0.02s)
                current_step += 1
                current_sim_time = current_step * rl_dt

                # 6. 打印状态
                if current_step % 50 == 0:
                    print(f"Time: {current_sim_time:.2f}s | Cmd: {current_command} | Vel_x: {env.base_lin_vel[0,0]:.2f}")

                # 7. 强制重绘
                if hasattr(env, 'viewer') and env.viewer is not None:
                    env.viewer.update()

                # 8. [修改 4] 1:1 挂钟时间同步 (核心限速器)
                # 如果推理和渲染的耗时小于 0.02s，则休眠补齐，确保肉眼观感与物理时间 1:1 匹配
                elapsed = time.perf_counter() - loop_start
                if elapsed < rl_dt:
                    time.sleep(rl_dt - elapsed)

            print("✅ 自动化测试序列执行完毕。")

    except KeyboardInterrupt:
        print(f"\n⏹️ 评估被手动中断。")


if __name__ == "__main__":
    main()

