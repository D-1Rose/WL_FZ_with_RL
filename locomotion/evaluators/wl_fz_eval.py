import argparse
import os
import pickle

import torch
from wl_fz_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(current_dir)
print(parent_dir)
sys.path.append(parent_dir)
from utils import gamepad
import copy


def plot_logs(log_data, dt=0.01):
    # 将 list 转为 numpy array
    time_steps = np.arange(len(log_data["cmd_lin_vel_x"])) * dt
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Evaluation Analysis', fontsize=16)
    
    # 子图 1: 线速度跟踪
    ax = axes[0, 0]
    ax.plot(time_steps, log_data["cmd_lin_vel_x"], 'r--', label='Command')
    ax.plot(time_steps, log_data["base_lin_vel_x"], 'b-', label='Actual')
    ax.set_title('Linear Velocity Tracking (X)')
    ax.set_ylabel('Velocity (m/s)')
    ax.legend()
    ax.grid(True)
    
    # 子图 2: 角速度跟踪
    ax = axes[0, 1]
    ax.plot(time_steps, log_data["cmd_ang_vel_yaw"], 'r--', label='Command')
    ax.plot(time_steps, log_data["base_ang_vel_yaw"], 'b-', label='Actual')
    ax.set_title('Angular Velocity Tracking (Yaw)')
    ax.grid(True)
    
    # 子图 3: Pitch 姿态稳定性
    ax = axes[1, 0]
    ax.plot(time_steps, np.rad2deg(log_data["pitch"]), 'g-', label='Pitch (deg)')
    ax.set_title('Body Pitch Angle')
    ax.set_ylabel('Degree')
    ax.axhline(y=20, color='r', linestyle=':', label='Warning Threshold')
    ax.axhline(y=-20, color='r', linestyle=':')
    ax.legend()
    ax.grid(True)
    
    # 子图 4: 动作平滑度 (Action Smoothness)
    ax = axes[1, 1]
    ax.plot(time_steps, log_data["action_smoothness"], 'k-', linewidth=0.5)
    ax.set_title('Action Change Rate (Smoothness)')
    ax.set_ylabel('|Action_t - Action_{t-1}|')
    ax.grid(True)

    # 子图 5: 参数演变 (以 Pitch Kp 和 Vel Kp 为例)
    # 注意：fuzzy_params 是 [N, 9] 的数组，你需要根据索引取值
    # 假设 index 5 是 pitch_kp, index 7 是 vel_kp
    fuzzy_data = np.array(log_data["fuzzy_params"])
    if len(fuzzy_data) > 0:
        ax = axes[2, 0]
        ax.plot(time_steps, fuzzy_data[:, 5], label='Pitch Kp')
        ax.plot(time_steps, fuzzy_data[:, 7], label='Vel Kp')
        ax.set_title('PID Parameters Adaptation')
        ax.legend()
        ax.grid(True)

    # 子图 6: 机械震动 (Joint Acc)
    ax = axes[2, 1]
    ax.plot(time_steps, log_data["dof_acc_mean"], 'm-', linewidth=0.5)
    ax.set_title('Mechanical Vibration (Joint Acc Mean)')
    ax.set_ylabel('rad/s^2')
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking-v15000")
    parser.add_argument("--ckpt", type=int, default=15000)
    # 新增：控制方式切换（0=原位置/速度控制，1=力矩控制）
    parser.add_argument("--control_mode", type=int, default=0, help="0: original control, 1: torque control")
    # 新增：PID参数（方便命令行调参，无需修改代码）
    parser.add_argument("--pos_kp", type=float, default=0, help="PID kp for position-to-torque")
    parser.add_argument("--pos_ki", type=float, default=0, help="PID ki for position-to-torque")
    parser.add_argument("--pos_kd", type=float, default=0, help="PID kd for position-to-torque")

    parser.add_argument("--vel_kp", type=float, default=1, help="PID kp for velocity-to-torque")
    parser.add_argument("--vel_ki", type=float, default=0.5, help="PID ki for velocity-to-torque")
    parser.add_argument("--vel_kd", type=float, default=0, help="PID kd for velocity-to-torque")
    args = parser.parse_args()

    # 初始化日志（记录对比数据）
    from torch.utils.tensorboard import SummaryWriter
    log_dir = f"./control_debug_{args.control_mode}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    step_count = [0]  # 全局步数计数器

    gs.init(backend=gs.cuda,logging_level="warning")
    gs.device="cuda:0"
    log_dir = f"/home/huang/wheel_leg/wheel_legged_genesis_new/locomotion/logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(f"/home/huang/wheel_leg/wheel_legged_genesis_new/locomotion/logs/{args.exp_name}/cfgs.pkl", "rb"))
    # env_cfg["simulate_action_latency"] = False
    terrain_cfg["terrain"] = True
    terrain_cfg["eval"] = "agent_eval_gym" #agent_eval_gym/agent_train_gym/circular
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
        # 新增参数：控制方式和PID参数
        control_mode=args.control_mode,  # 0=原控制，1=力矩控制
        pos_kp=args.pos_kp,
        pos_ki=args.pos_ki,
        pos_kd=args.pos_kd,
        vel_kp=args.vel_kp,
        vel_ki=args.vel_ki,
        vel_kd=args.vel_kd,
        writer=writer,  # 传入TensorBoard日志器
        step_count=step_count  # 传入步数计数器（列表形式）
    )


    
    print(reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    #jit
    model = copy.deepcopy(runner.alg.actor_critic.actor).to('cpu')
    torch.jit.script(model).save(log_dir+"/policy.pt")
    # 加载模型进行测试
    print("\n--- 模型加载测试 ---")
    try:
        loaded_policy = torch.jit.load(log_dir + "/policy.pt")
        loaded_policy.eval() # 设置为评估模式
        loaded_policy.to('cuda')
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()
    obs, _ = env.reset()   # 重置环境，获取初始观测值
    pad = gamepad.control_gamepad(command_cfg,[1.3,1.0,1.3,0.0125])  # 初始化控制设备（手柄/键盘）

# # =================================================================================
#     # [新增] 定义分段测试指令序列
#     # 格式: (开始时间, 结束时间, [vel_x, vel_y, ang_vel_yaw, height_target])
#     # 注意：时间单位为秒
#     # =================================================================================
#     test_schedule = [
#         (0.0,  2.0,  [0.0, 0.0, 0.0, 0.28]),   # 阶段1: 0-2s 原地静止 (让机器人稳定下来)
#         (2.0,  7.0,  [-0.3, 0.0, 0.0, 0.28]),   # 阶段2: 2-6s 直线加速 (测试速度响应)
#         (7.0,  12.0,  [-0.6, 0.0, 0.0, 0.28]),   # 阶段3: 6-8s 急刹 
#         (12.0,  17.0, [0.0, 0.0, 0.0, 0.28]),   # 阶段4: 8-10s 停 
#         (17.0, 22.0, [0.3, 0.0, 0.0, 0.28]),   # 阶段5: 10-12s 原地静止
#         (22.0, 27.0, [0.6, 0.0, 0.0, 0.28]),   # 阶段6: 15-20s 直线加速
#         (27.0, 32.0, [0.0, 0.0, 0.0, 0.28]),   # 阶段7: 20-22s 停
#     ]
    
#     # 计算总测试时长
#     max_test_time = test_schedule[-1][1]
#     print(f"\n🚀 开始自动化分段测试，总时长: {max_test_time}s")

# # 获取仿真步长 (通常是 0.01s 或 0.02s)
#     sim_dt = env.dt 
#     current_sim_time = 0.0
#     current_step = 0

#     with torch.no_grad():
#         while current_sim_time < max_test_time: # 当时间结束时自动退出
            
#             # 1. 策略推理
#             # actions = policy(obs)
#             actions = loaded_policy(obs)
            
#             # 2. 环境步进
#             obs, _, rews, dones, infos = env.step(actions)
            
#             # 3. [核心修改] 根据当前时间查找指令
#             current_command = [0.0, 0.0, 0.0, 0.28] # 默认指令
            
#             for (start, end, cmd) in test_schedule:
#                 if start <= current_sim_time < end:
#                     current_command = cmd
#                     break
            
#             # 4. 发送指令给环境
#             # env.set_commands 需要接收 tensor 或 list
#             # 假设你只测试第0个环境
#             env.set_commands(0, current_command)
            
#             # 5. 更新时间
#             current_step += 1
#             current_sim_time = current_step * sim_dt
            
#             # 可选：打印进度条或当前阶段
#             if current_step % 50 == 0:
#                 print(f"Time: {current_sim_time:.2f}s | Cmd: {current_command}")

#     print("✅ 测试结束，已完成所有指令序列。")


    with torch.no_grad():
        
        while True:
        # for i in range(10):
            # actions = policy(obs)
            actions = loaded_policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            comands,reset_flag = pad.get_commands()   # 读取手柄/键盘输入
            env.set_commands(0,comands)
            if reset_flag:
                env.reset()
            
            


if __name__ == "__main__":
    main()

