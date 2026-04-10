import time
import math
import argparse
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# 导入你自己的控制器模块
from mj_keyboard import KeyboardCommander
from mj import VMC, LQR_Controller

def main():
    # --- 0. 参数解析器 (用于灵活配置测试条件) ---
    parser = argparse.ArgumentParser(description="轮腿机器人底层参数对比测试脚本")
    parser.add_argument("--domain", type=str, default="15°", help="当前测试的论域大小 (用于标题)")
    parser.add_argument("--gain", type=str, default="1.0", help="当前测试的 RL 增益/比例因子 (用于标题)")
    parser.add_argument("--auto", action="store_true", help="开启自动化测试模式 (屏蔽键盘)")
    parser.add_argument("--duration", type=float, default=10, help="自动化测试总时长 (秒)")
    parser.add_argument("--target_v", type=float, default=1.0, help="自动化测试的目标速度 (m/s)")
    args = parser.parse_args()

    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path("/home/huang/wheel_leg/wheel_legged_genesis_new/assets/description/urdf/scence.xml")
    data = mujoco.MjData(model)
    
    # 2. 实例化控制器
    vmc_ctrl = VMC(model)
    lqr_ctrl = LQR_Controller(model)    
    
    # 3. 实例化键盘控制器
    cmd = KeyboardCommander(v_step=0.1, yaw_step_deg=10.0)

    # --- 数据记录器 (Data Logger) ---
    log_time = []
    log_target_v = []
    log_real_v = []
    log_pitch_deg = []
    log_pitch_dot = []
    log_theta_pitch_deg = [] # [新增] 内环俯仰误差

    print("\n" + "="*60)
    if args.auto:
        print(f"🤖 自动化测试已启动 | 时长: {args.duration}s | 目标速度: {args.target_v} m/s")
    else:
        print("🎮 手动键盘模式已启动 | 请使用键盘控制机器人。")
    print(f"📊 记录标签 -> 论域: {args.domain} | 增益: {args.gain}")
    print("="*60 + "\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        model.opt.timestep = 0.002
        
        # --- 设置渲染解频参数 ---
        render_fps = 60  
        physics_steps_per_render = int((1.0 / render_fps) / model.opt.timestep) 
        
        step = 0
        sim_time = 0.0
        global_start_time = time.perf_counter()

        while viewer.is_running():
            step_start = time.perf_counter()

            # --- 物理与控制高频循环 (500Hz) ---
            for _ in range(physics_steps_per_render):
                step += 1
                sim_time = step * model.opt.timestep
                
                # --- [新增] 指令生成逻辑 (区分自动与手动) ---
                if args.auto:
                    # 自动化模式：前1秒平滑加速，防止起步瞬间力矩过爆
                    if sim_time < 1.0:
                        target_v = args.target_v * (sim_time / 1.0)
                    else:
                        target_v = args.target_v
                    target_yaw = 0.0
                else:
                    # 手动键盘模式
                    target_v, target_yaw = cmd.get_command()

                # 注入指令到 LQR 控制器
                lqr_ctrl.velocity_d = target_v
                lqr_ctrl.yaw_d = target_yaw

                # A. 统一更新状态
                lqr_ctrl.update_imu_data(data)
                lqr_ctrl.update_joint_states(data)
                vmc_ctrl.update_states(data)

                # B. 统一计算控制
                vmc_ctrl.vmc(data)     
                lqr_ctrl.balance(data) 

                # C. 物理步进
                mujoco.mj_step(model, data)
                
                # --- D. 高频数据采集 ---
                log_time.append(sim_time)
                log_target_v.append(target_v)
                log_real_v.append(lqr_ctrl.robot_x_velocity)
                log_pitch_deg.append(lqr_ctrl.pitch * (180.0 / math.pi)) 
                log_pitch_dot.append(lqr_ctrl.pitch_dot)
                # 记录内环误差 (确保 LQR_Controller 中有 theta_pitch 属性)
                log_theta_pitch_deg.append(lqr_ctrl.theta_pitch * (180.0 / math.pi))

            # --- 画面渲染低频循环 (60Hz) ---
            viewer.sync()

            # --- [新增] 自动化测试超时退出 ---
            if args.auto and sim_time >= args.duration:
                print(f"\n🏁 达到设定的测试时长 ({args.duration}s)，自动结束仿真。")
                viewer.close() # 触发关闭，跳出循环
                break

            # --- 时间绝对对齐检查 ---
            if step % int(0.5 / model.opt.timestep) == 0:
                real_time_elapsed = time.perf_counter() - global_start_time
                time_diff = real_time_elapsed - sim_time
                print(f"⏱️ 仿真时间: {sim_time:.3f}s | 偏差: {time_diff:+.3f}s | V_err: {target_v - lqr_ctrl.robot_x_velocity:+.2f} | θ_pitch: {lqr_ctrl.theta_pitch * 180 / math.pi:+.2f}°")

            # --- 挂钟时间真实同步 ---
            expected_time = physics_steps_per_render * model.opt.timestep
            elapsed = time.perf_counter() - step_start
            
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

    # ====================================================
    # 自动绘制性能曲线
    # ====================================================
    print("\n✅ 正在生成性能评估图表...")
    
    # 构建图表标题
    title_str = f"Test Result -> Domain:{args.domain}    Gain:{args.gain}"
    plot_results(log_time, log_target_v, log_real_v, log_pitch_deg, log_pitch_dot, log_theta_pitch_deg, title_str)


def plot_results(time_data, target_v, real_v, pitch_deg, pitch_dot, theta_pitch_deg, title_str):
    """重构后的三维数据画图函数"""
    plt.style.use('bmh') 
    # [修改] 增加为 3 个子图，并适当加高画布
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 注入定制化大标题
    fig.suptitle(title_str, fontsize=16, fontweight='bold', color='navy')
    
    # 1. 速度跟踪图
    ax1.plot(time_data, target_v, 'r--', label='Target Velocity (m/s)', linewidth=2)
    ax1.plot(time_data, real_v, 'b-', label='Real Velocity (m/s)', linewidth=1.5)
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Outer Loop: Velocity Tracking')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # 2. 绝对俯仰角与角速度图
    ax2.plot(time_data, pitch_deg, 'g-', label='Absolute Pitch (deg)', linewidth=1.5)
    ax2_vel = ax2.twinx()
    ax2_vel.plot(time_data, pitch_dot, 'k-', alpha=0.3, label='Pitch Dot (rad/s)')
    
    ax2.set_ylabel('Pitch Angle (deg)', color='g')
    ax2_vel.set_ylabel('Pitch Dot (rad/s)', color='k')
    ax2.set_title('Base Attitude & Stability')
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_vel.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # 3. [新增] 核心内环跟踪误差图 (Theta Pitch)
    # 这张图最能反映模糊控制器的抵抗能力：遇到扰动尖峰越小、回零越快，说明控制器越强
    ax3.plot(time_data, theta_pitch_deg, 'm-', label='Theta Pitch Error (deg)', linewidth=1.5)
    ax3.axhline(0, color='black', linestyle=':', linewidth=1) # 增加 0 刻度基准线
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error Angle (deg)', color='m')
    ax3.set_title('Inner Loop: Theta Pitch (Tracking Error)')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    plt.tight_layout()
    # 调整布局以为 suptitle 留出空间
    plt.subplots_adjust(top=0.92) 
    plt.show()

if __name__ == "__main__":
    main()