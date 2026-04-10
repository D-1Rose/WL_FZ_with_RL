import time
import math
import numpy as np
import torch
import mujoco
import mujoco.viewer
from transforms3d import euler, quaternions
import argparse
import os
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 引入你的原始控制器
from controllers.Offset_mj_controller import LQR_Controller, VMC, PIDController
from mj_keyboard import KeyboardCommander

# --- 配置参数 (必须与 Genesis 训练配置完全一致) ---
class CFG:
    # 1. 观测标准化系数 (必须与 FZ_PID_train.py 的 obs_scales 严格一致)
    scales = {
        "lin_vel": 1.0,
        "ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,       # [修改] 匹配 Genesis 训练配置
        "height": 5.0          # [修改] 匹配 Genesis 训练配置
    }

    # 2. 默认关节角度 (保持不变)
    default_dof_pos = np.array([
        -0.28, 0.3385,  # L1, L2
        0.28, -0.3385,  # R1, R2
        0.0, 0.0        # L3, R3
    ])
    
    # 3. [彻底修改] 残差动作配置
    # 删掉之前的 action_mapping 列表，改用物理限幅和系数
    MAX_PITCH_OFFSET = 0.25     # [新增] 对应 Genesis 环境中的 0.1 弧度限制
    action_smooth_factor = 0.0  # [新增] 对应 Genesis 的平滑因子
    
    # 4. [核心修改] 维度对齐
    num_actions = 1  # [修改] 只输出 1 维俯仰补偿 (rl_pitch_offset)
    num_obs = 26     # [修改] 单帧维度改为 26 (3+3+3+4+4+6+1+2)
    history_len = 5  # 保持 5 帧历史
    
    # 5. 设备与频率
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim_dt = 0.002     # 物理仿真 500Hz
    control_decimation = 10 # RL 决策 50Hz (必须与 Genesis 的 decimation=10 一致)

class RL_Adapter:
    def __init__(self, model_path, model):
        self.device = CFG.device
        print(f"🚀 加载残差策略模型: {model_path}")
        self.policy = torch.jit.load(model_path).to(self.device)
        self.policy.eval()
        
        # 历史观测 Buffer: (1, 5, 26) -> 必须对齐最新的 26 维
        self.obs_history = torch.zeros((1, CFG.history_len, CFG.num_obs), device=self.device)
        
        # 上一帧动作 (用于构造观测中的 actions)
        self.last_actions = torch.zeros((1, CFG.num_actions), device=self.device)
        
        # 关节顺序映射 (Genesis: L1, L2, R1, R2, L3, R3)
        self.joint_names = ["L1_joint", "L2_joint", "R1_joint", "R2_joint", "L3_joint", "R3_joint"]
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]

    def get_observation(self, data, lqr_ctrl, target_vel_x, target_yaw_dot):
        """
        构建与 Genesis 纯净版完全一致的 26 维观测向量
        """
        qpos = data.qpos
        qvel = data.qvel
        quat = qpos[3:7]  # [w, x, y, z]
        
        # 坐标系转换
        R = quaternions.quat2mat(quat) 
        v_body = R.T @ qvel[0:3]
        w_body = qvel[3:6]
        projected_gravity = R.T @ np.array([0.0, 0.0, -1.0])
        
        # 关节数据 (按 Genesis 顺序)
        dof_pos = np.array([qpos[i] for i in self.joint_ids])
        dof_vel = np.array([qvel[i] for i in self.joint_ids])
        
        # ================== 严格按照 Genesis FZ_PID_env_c.py 的顺序拼接 ==================
        # 【注意】全部采用原始物理量，不乘任何 scale，否则会导致数据落入神经网络的死区
        
        obs_lin_vel = torch.tensor(v_body, device=self.device).float()              # 3
        obs_ang_vel = torch.tensor(w_body, device=self.device).float()              # 3
        obs_gravity = torch.tensor(projected_gravity, device=self.device).float()     # 3
        # commands: [vx, vy, yaw_dot, height]
        obs_commands = torch.tensor([target_vel_x, 0.0, target_yaw_dot, 0.28], device=self.device).float() # 4
        # dof_pos_error (4 legs)
        obs_dof_pos = torch.tensor(dof_pos[0:4] - CFG.default_dof_pos[0:4], device=self.device).float()   # 4
        obs_dof_vel = torch.tensor(dof_vel, device=self.device).float()             # 6
        obs_actions = self.last_actions.squeeze(0)                                  # 1
        obs_pitch = torch.tensor([lqr_ctrl.pitch, lqr_ctrl.pitch_dot], device=self.device).float()     # 2
        
        # 拼接 (3+3+3+4+4+6+1+2 = 26维)
        current_obs = torch.cat([
            obs_lin_vel, obs_ang_vel, obs_gravity, obs_commands, 
            obs_dof_pos, obs_dof_vel, obs_actions, obs_pitch
        ]).unsqueeze(0) 
        
        return current_obs

    def step(self, data, lqr_ctrl, target_vel, target_yaw):
        """
        执行推理并返回 Pitch 偏置 (单位：弧度)
        """
        slice_obs = self.get_observation(data, lqr_ctrl, target_vel, target_yaw)
        
        # 构造时序输入 (1, 156)
        policy_input = torch.cat([self.obs_history, slice_obs.unsqueeze(1)], dim=1).view(1, -1)
        
        with torch.no_grad():
            # 神经网络输出动作 [-1, 1]
            raw_action = torch.tanh(self.policy(policy_input))
        
        # 更新历史 Buffer
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :].clone()
        self.obs_history[:, -1, :] = slice_obs

        # 【平滑逻辑】必须与 FZ_PID_env_c.py 的 0.8 对齐
        self.last_actions = CFG.action_smooth_factor * self.last_actions + (1 - CFG.action_smooth_factor) * raw_action

        # 【映射逻辑】[-1, 1] -> [-0.1, 0.1] rad (约 ±5.7度)
        pitch_offset = self.last_actions[0, 0].item() * CFG.MAX_PITCH_OFFSET
        
        return -pitch_offset*0.5

# --- Main 函数 ---
def main():
    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="MuJoCo Eval with TensorBoard & RL Switch (Residual RL Version)")
    parser.add_argument("--RL", action="store_true", help="开启 RL 俯仰角残差补偿")
    parser.add_argument("--log_name", type=str, default="experiment", help="TensorBoard 日志名称前缀")
    args = parser.parse_args()

    # 1. 加载模型
    xml_path = "/home/huang/wheel_leg/wheel_legged_genesis_new/assets/description/urdf/scence.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = CFG.sim_dt # 确保 500Hz 步进

    # =================================================================
    # [参数配置区] 物理与逻辑设定 (保持与你的设置一致)
    # =================================================================
    TARGET_V_FINAL = 0.5     # 目标速度 (m/s)
    MAX_ACCEL = 2.5         # 期望加速度 (m/s^2)
    WARM_UP_TIME = 1      # 缓冲时间 (s)
    FINISH_LINE_X = 7.5      # 自动停止的终点线 (m)
    
    max_dv = MAX_ACCEL * model.opt.timestep 
    
    print(f"\n🚀 残差 RL 自动评估模式已启动:")
    print(f"   目标速度: {TARGET_V_FINAL} m/s | 设定加速度: {MAX_ACCEL} m/s²")
    print(f"   终点线位置: {FINISH_LINE_X} m\n")

    # 2. 初始化 TensorBoard 与日志
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode_str = "ResidualRL" if args.RL else "PurePID"
    log_dir = f"logs/mujoco_comparison/{args.log_name}_{mode_str}_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard 日志目录: {log_dir}")
    print(f"🤖 当前模式: [{'RL 残差开启' if args.RL else '纯 PID 运行'}]\n")

    # 3. 实例化控制器与 RL 适配器
    vmc_ctrl = VMC(model)
    lqr_ctrl = LQR_Controller(model)
    rl_adapter = None
    
    if args.RL:
        # [请确保此路径指向你最新的残差版本模型]
        policy_path = "/home/huang/wheel_leg/wheel_legged_genesis_new/locomotion/logs/offset_20_eval/2026-03-13_19-59-43_ckpt300/policy_fused.pt"
        try:
            rl_adapter = RL_Adapter(policy_path, model)
        except Exception as e:
            print(f"❌ 无法加载残差模型: {e}")
            return

    # 4. CSV 文件初始化 (增加 pitch_offset 列)
    csv_filename = f"robot_log_{mode_str}_{timestamp}.csv"
    csv_path = os.path.join(log_dir, csv_filename)
    header = ['time', 'target_v', 'real_v', 'vel_error', 'pitch', 'pitch_dot', 'pitch_offset_deg', 'pos_x']
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

    # 5. 初始化状态变量
    smoothed_v = 0.0
    pitch_offset = 0.0 # RL 输出的补偿量
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        step_counter = 0
        prev_sim_time = 0.0
        render_fps = 60 
        physics_steps_per_render = int((1.0 / render_fps) / model.opt.timestep) 

        while viewer.is_running():
            step_start = time.perf_counter() 
            
            for _ in range(physics_steps_per_render):
                # --- 1. 状态获取与终点判定 ---
                current_x = data.body('base_link').xpos[0]
                real_vel = lqr_ctrl.robot_x_velocity 
                
                if current_x >= FINISH_LINE_X:
                    print(f"\n🏁 [任务完成] 机器人成功抵达终点 ({current_x:.2f}m)")
                    viewer.close() 
                    break 

                # --- 2. 处理仿真重置 ---
                if data.time < prev_sim_time:
                    lqr_ctrl.reset()
                    smoothed_v = 0.0
                    pitch_offset = 0.0
                    lqr_ctrl.rl_pitch_offset = 0.0 # 核心清零
                    if args.RL and rl_adapter is not None:
                        rl_adapter.obs_history.zero_()
                        rl_adapter.last_actions.zero_()
                
                # --- 3. 指令平滑斜坡 (Warm-up & Accel) ---
                current_target = TARGET_V_FINAL if data.time > WARM_UP_TIME else 0.0
                v_diff = current_target - smoothed_v
                v_diff = np.clip(v_diff, -max_dv, max_dv)
                smoothed_v += v_diff
                
                # --- 4. 控制器预更新 ---
                target_v = smoothed_v
                lqr_ctrl.velocity_d = smoothed_v
                lqr_ctrl.yaw_d = 0.0
                lqr_ctrl.update_imu_data(data)
                lqr_ctrl.update_joint_states(data)
                vmc_ctrl.update_states(data)

                # --- 5. [核心修改] RL 俯仰残差介入 ---
                if args.RL and (step_counter % CFG.control_decimation == 0):
                    # 现在 step 函数返回的是计算好的 pitch_offset (float)
                    pitch_offset = rl_adapter.step(data, lqr_ctrl, target_v, 0.0)
                
                # 直接注入底层控制器的残差接口
                lqr_ctrl.rl_pitch_offset = pitch_offset

                # --- 6. 执行物理计算 ---
                vmc_ctrl.vmc(data)
                lqr_ctrl.balance(data)
                mujoco.mj_step(model, data)

                # --- 7. 数据记录 (TensorBoard) ---
                if step % 10 == 0:
                    vel_error = target_v - real_vel
                    offset_deg = lqr_ctrl.rl_pitch_offset 
                    
                    writer.add_scalar("Tracking/Vel_Target", target_v, step_counter)
                    writer.add_scalar("Tracking/Vel_Real", real_vel, step_counter)
                    writer.add_scalar("Tracking/Vel_Error", vel_error, step_counter)
                    
                    writer.add_scalar("Attitude/Pitch_deg", lqr_ctrl.pitch , step_counter)
                    writer.add_scalar("Attitude/Roll_deg", lqr_ctrl.roll , step_counter)
                    
                    # [新增记录] 观察残差补偿力度
                    writer.add_scalar("RL_Actions/Pitch_Offset_Deg", offset_deg, step_counter)

                # --- 8. CSV 与终端打印 ---
                if step_counter % 10 == 0:
                    csv_writer.writerow([
                        f"{data.time:.4f}", f"{target_v:.4f}", f"{real_vel:.4f}", 
                        f"{target_v - real_vel:.4f}", f"{lqr_ctrl.pitch:.4f}", 
                        f"{lqr_ctrl.pitch_dot:.4f}", f"{lqr_ctrl.rl_pitch_offset * 180/math.pi:.4f}",
                        f"{current_x:.4f}"
                    ])

                if step % 100 == 0: # 降低打印频率
                    print(f"T: {data.time:.2f}s | X: {current_x:.2f}m | RL Offset: {lqr_ctrl.rl_pitch_offset*180/math.pi:+.2f}°")

                prev_sim_time = data.time
                step_counter += 1
                step += 1
            
            if not viewer.is_running():
                break

            viewer.sync()
            expected_time = physics_steps_per_render * model.opt.timestep
            elapsed = time.perf_counter() - step_start
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

    csv_file.close()
    print(f"\n✅ 评估结束。日志文件已保存至: {csv_path}")

if __name__ == "__main__":
    main()