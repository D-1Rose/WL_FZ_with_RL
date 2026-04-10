import argparse
import os
import sys
import time
import math
import torch
import genesis as gs
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.FZ_PID_env_c import WheelLeggedEnv
from trainers.FZ_PID_train import get_cfgs

def plot_results(time_data, target_v, real_v, pitch_deg, pitch_dot):
    """绘制控制性能数据图，用于精准调参"""
    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. 速度跟踪图
    ax1.plot(time_data, target_v, 'r--', label='Target Velocity (m/s)', linewidth=2)
    ax1.plot(time_data, real_v, 'b-', label='Real Velocity (m/s)', linewidth=1.5)
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Genesis: Velocity Tracking Performance')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 俯仰角姿态图
    ax2.plot(time_data, pitch_deg, 'g-', label='Pitch Angle (deg)', linewidth=1.5)
    
    ax2_vel = ax2.twinx()
    ax2_vel.plot(time_data, pitch_dot, 'k-', alpha=0.3, label='Pitch Dot (rad/s)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Pitch Angle (deg)', color='g')
    ax2_vel.set_ylabel('Pitch Dot (rad/s)', color='k')
    ax2.set_title('Genesis: Attitude Stability')
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_vel.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Genesis PID 调参验证工具")
    parser.add_argument("--use_keyboard", action="store_true", default=True, help="默认使用键盘控制")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 Genesis PID 调参模式启动！")
    print("   操作说明：")
    print("   1. 在弹出的仿真窗口中使用键盘控制机器人。")
    print("   2. 观察终端输出的 '仿真时间 vs 现实时间' 确保 1:1 对齐。")
    print("   3. 在终端按下 Ctrl+C 结束测试，将自动弹出 PID 性能分析图表！")
    print("="*60 + "\n")
    
    # ===============================================================
    # 强制使用 CPU 后端以消除内核启动延迟，确保单体测试绝对流畅
    # ===============================================================
    gs.init(backend=gs.cpu, logging_level="warning") 
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
    
    # 简化地形，专心调参
    terrain_cfg["terrain"] = False 
    
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
        device="cpu"  # 强制 CPU
    )
    
    pad = None
    if args.use_keyboard:
        try:
            sys.path.append("/home/zhjkoi/wheel_leg_paper/wheel_legged_genesis/locomotion")
            from utils import gamepad
            commands_scale = [0.1, 0.5, 1.57, 0.28]
            pad = gamepad.control_gamepad(command_cfg, commands_scale)
        except Exception as e:
            print(f"键盘控制初始化失败: {e}")

    obs, _ = env.reset()
    
    rl_dt = env.dt * env.decimation  # 0.02s
    current_sim_time = 0.0   
    current_step = 0         
    
    # --- 数据记录器 (Data Logger) ---
    log_time = []
    log_target_v = []
    log_real_v = []
    log_pitch_deg = []
    log_pitch_dot = []

    global_start_time = time.perf_counter()

    try:
        with torch.no_grad():
            while True: # 依赖 Ctrl+C 退出
                loop_start = time.perf_counter()

                # 获取指令
                current_command = [0.0, 0.0, 0.0, 0.28]
                if pad is not None:
                    commands, reset_flag = pad.get_commands()
                    current_command = commands
                    if reset_flag:
                        env.reset()
                        current_step = 0
                        current_sim_time = 0.0
                        log_time.clear(); log_target_v.clear(); log_real_v.clear()
                        log_pitch_deg.clear(); log_pitch_dot.clear()
                        global_start_time = time.perf_counter()
                        print("\n🔄 环境与数据已重置\n")
                        continue
                
                env.set_commands([0], current_command)

                # 物理步进 (纯 CPU 模式极快)
                actions = torch.zeros((1, env.num_actions), device="cpu")
                obs, _, rews, dones, infos = env.step(actions)
                
                current_step += 1
                current_sim_time = current_step * rl_dt

                # --- 提取核心指标 ---
                target_v = current_command[0]
                real_v = env.base_lin_vel[0, 0].item()
                pitch_rad = env.base_euler[0, 1].item()
                pitch_deg = pitch_rad 
                pitch_dot = env.base_ang_vel[0, 1].item()

                # --- 记录数据 ---
                log_time.append(current_sim_time)
                log_target_v.append(target_v)
                log_real_v.append(real_v)
                log_pitch_deg.append(pitch_deg)
                log_pitch_dot.append(pitch_dot)

                # --- 控制台打印与时间同步检查 ---
                if current_step % int(0.5 / rl_dt) == 0:
                    real_time_elapsed = time.perf_counter() - global_start_time
                    time_diff = real_time_elapsed - current_sim_time
                    print(f"⏱️ 仿真时间: {current_sim_time:.2f}s | 现实时间: {real_time_elapsed:.2f}s | 偏差: {time_diff:+.3f}s")
                    print(f"   📊 目标速度: {target_v:.2f} | 实际速度: {real_v:.2f} | Pitch: {pitch_deg:.2f}°")
                    print("-" * 60)
                
                # --- 挂钟限速锁 (1:1 同步核心) ---
                elapsed = time.perf_counter() - loop_start
                if elapsed < rl_dt:
                    time.sleep(rl_dt - elapsed)
                
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⏹️  测试被手动中断，正在生成性能评估图表...")
        print("="*60)
    finally:
        # 生成图表
        if len(log_time) > 0:
            plot_results(log_time, log_target_v, log_real_v, log_pitch_deg, log_pitch_dot)

if __name__ == "__main__":
    main()