import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_mj_performance(csv_path):
    # 1. 读取数据
    if not os.path.exists(csv_path):
        print(f"❌ 错误：文件不存在 {csv_path}")
        return

    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 2. 设置绘图风格
    plt.style.use('seaborn-v0_8-deep') # 或者使用 'bmh'
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- 子图 1: 速度跟踪对比 ---
    ax1.plot(df['time'], df['target_v'], 'r--', label='Target Velocity (m/s)', linewidth=1.5)
    ax1.plot(df['time'], df['real_v'], 'b-', label='Real Velocity (m/s)', linewidth=1.2)
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('MuJoCo: Velocity Tracking Performance')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- 子图 2: 俯仰角姿态与 RL 补偿 ---
    # 转换为角度便于观察 (如果 CSV 里已经是度数则直接画)
    pitch_deg = df['pitch'] * 180 / 3.14159 if df['pitch'].abs().max() < 2 else df['pitch']
    
    ax2.plot(df['time'], pitch_deg, 'g-', label='Base Pitch (deg)', linewidth=1.2)
    # 绘制 RL 注入的残差补偿量
    ax2.plot(df['time'], df['pitch_offset_deg'], 'm--', label='RL Pitch Offset (deg)', linewidth=1.5)
    
    ax2.set_ylabel('Angle (deg)')
    ax2.set_title('Base Attitude & RL Intervention')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- 子图 3: 速度误差与位移 ---
    ax3.plot(df['time'], df['vel_error'], 'k-', label='Velocity Error', alpha=0.6)
    ax3.fill_between(df['time'], df['vel_error'], 0, color='gray', alpha=0.2)
    ax3.set_ylabel('Error (m/s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Velocity Error over Time')
    
    # 在右侧增加一个轴显示位移
    ax3_pos = ax3.twinx()
    ax3_pos.plot(df['time'], df['pos_x'], 'c:', label='Position X (m)', linewidth=2)
    ax3_pos.set_ylabel('Position (m)')
    
    ax3.legend(loc='upper left')
    ax3_pos.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 自动保存图片到数据所在目录
    save_path = csv_path.replace('.csv', '.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表已生成并保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 在这里填入你的日志文件路径
    log_file = "logs/mujoco_comparison/experiment_ResidualRL_2026-03-15_23-18-39/robot_log_ResidualRL_2026-03-15_23-18-39.csv"
    plot_mj_performance(log_file)