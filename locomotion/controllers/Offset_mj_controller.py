import math
from transforms3d import euler
import numpy as np
import time
import mujoco
import mujoco.viewer

# --- PID 工具类 ---
class PIDController:
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        # 1. 计算积分 (带限幅)
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
            
        # 2. 计算微分
        derivative = (error - self.prev_error) / dt
        
        # 3. 计算输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 4. 更新历史
        self.prev_error = error
        
        # [新增] 5. 总输出限幅
        if self.output_limit is not None:
            output = max(min(output, self.output_limit), -self.output_limit)
            
        return output
    # [新增] 重置函数：清除历史记忆
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

# --- LQR 控制器 (保持原有函数名和核心逻辑) ---
class LQR_Controller:
    def __init__(self, model):
        self.model = model
        self.dt = 0.002
        
        # 内部状态变量
        self.base_quat = [1, 0, 0, 0]
        self.left_wheel_velocity = 0.0
        self.left_wheel_position = 0.0        
        self.right_wheel_velocity = 0.0
        self.right_wheel_position = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.pitch_dot = 0.0
        self.yaw = 0.0
        self.yaw_d = 0.0
        self.yaw_dot = 0.0
        self.theta_r1 = 0; self.theta_r2 = 0; self.theta_r3 = 0; self.theta_rb = 0
        self.theta_l1 = 0; self.theta_l2 = 0; self.theta_l3 = 0; self.theta_lb = 0
        self.robot_x_velocity = 0
        self.robot_x_position = 0.0
        self.velocity_d = 0.0
        self.theta_pitch = 0.0
        # 在 __init__ 中添加 rl_pitch_offset
        self.rl_pitch_offset = 0.0
        # PID 初始化
        #串级
        # self.pitch_d = -0.1
        # self.command_yaw = PIDController(kp=2, ki=0.01, kd=0.5, output_limit=50.0) 
        #                                     # kp=2, ki=0.01, kd=0.5
        # self.command_pitch = PIDController(kp=32.5, ki=0.1, kd=0.5, output_limit=100.0) # 平衡是大头
        #                                     # kp=25, ki=0, kd=0
        # self.command_velocity = PIDController(kp=1, ki=0.05, kd=0, output_limit=100.0) # 速度不要抢占太多力矩
        #                                     # kp=1, ki=0, kd=0

        # 并级
        self.command_yaw = PIDController(kp=1.5, ki=0.01, kd=0.5, output_limit=50.0) 
                                            # kp=2, ki=0.01, kd=0.5
        self.command_pitch = PIDController(kp=15, ki=0.0, kd=2.0, output_limit=100.0) # 平衡是大头
                                            # kp=25, ki=0, kd=0
        self.command_velocity = PIDController(kp=0.45, ki=0.001, kd=0, output_limit=100.0) # 速度不要抢占太多力矩
                                            # kp=1, ki=0, kd=0

        # 动态获取 ID
        self.joint_ids = {
            "left_hip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L1_joint"),
            "left_knee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L2_joint"),
            "left_wheel": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L3_joint"),
            "right_hip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R1_joint"),
            "right_knee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R2_joint"),
            "right_wheel": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R3_joint"),
        }
        self.actuator_ids = {
            "left_hip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L1_joint_ctrl"),
            "left_knee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L2_joint_ctrl"),
            "left_wheel": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L3_joint_ctrl"),
            "right_hip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R1_joint_ctrl"),
            "right_knee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R2_joint_ctrl"),
            "right_wheel": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R3_joint_ctrl"),
        }

        # [新增] 初始化可变的模糊控制参数 
        self.fuzzy_params = {
            "gamma_d_g": 75 * math.pi / 180,
            "eth": 0.06,
            "a": 0.1,
            "k_out": 1.0
        }

    def update_joint_states(self, data):
        # 提取位置
        self.left_wheel_position = data.qpos[self.joint_ids["left_wheel"]]
        self.right_wheel_position = data.qpos[self.joint_ids["right_wheel"]]
        self.left_hip_position = data.qpos[self.joint_ids["left_hip"]]
        self.right_hip_position = data.qpos[self.joint_ids["right_hip"]]
        self.left_knee_position = data.qpos[self.joint_ids["left_knee"]]
        self.right_knee_position = data.qpos[self.joint_ids["right_knee"]]

        # 提取速度
        self.left_wheel_velocity = data.qvel[self.joint_ids["left_wheel"]]
        self.right_wheel_velocity = data.qvel[self.joint_ids["right_wheel"]]
        self.left_knee_velocity = data.qvel[self.joint_ids["left_knee"]]
        self.right_knee_velocity = data.qvel[self.joint_ids["right_knee"]]
        self.left_hip_velocity = data.qvel[self.joint_ids["left_hip"]]
        self.right_hip_velocity = data.qvel[self.joint_ids["right_hip"]]

        # 动力学中间变量计算 (保留你的逻辑)
        # 右腿
        self.theta_r1 = self.right_wheel_position
        self.theta_r2 = -1*(95/180 * math.pi - self.right_knee_position)
        self.theta_r3 = self.right_hip_position - 20/180 * math.pi 
        self.theta_rb = -1*self.theta_r2 - self.theta_r3 - math.pi/2 + self.pitch
        self.theta_r1_dot = self.right_wheel_velocity
        self.theta_r2_dot = self.right_knee_velocity
        self.theta_r3_dot = self.right_hip_velocity
        self.theta_rb_dot = -self.theta_r2_dot - self.theta_r3_dot + self.pitch_dot

        # 左腿
        self.theta_l1 = -self.left_wheel_position
        self.theta_l2 = -1*(95/180 * math.pi + self.left_knee_position)
        self.theta_l3 = -self.left_hip_position - 20/180 * math.pi
        self.theta_lb = -1*self.theta_l2 - self.theta_l3 - math.pi/2 + self.pitch
        self.theta_l1_dot = -self.left_wheel_velocity
        self.theta_l2_dot = -self.left_knee_velocity
        self.theta_l3_dot = -self.left_hip_velocity
        self.theta_lb_dot = -self.theta_l2_dot - self.theta_l3_dot + self.pitch_dot

        # [严重错误修正] 
        # 原代码: self.robot_x_velocity = data.qvel[0]
        # 修改为: 计算机器人机身坐标系下的纵向速度
        
        v_global_x = data.qvel[0]
        v_global_y = data.qvel[1]
        v_global_z = data.qvel[2]
        # 旋转矩阵投影：v_body = v_x * cos(yaw) + v_y * sin(yaw)
        # 这里的 self.yaw 必须是最新的 (所以 update_imu_data 最好在 update_joint_states 之前调用)
        self.robot_x_velocity = v_global_x * math.cos(self.yaw) + v_global_y * math.sin(self.yaw)
        self.robot_x_position = (self.left_wheel_position - self.right_wheel_position) * 0.1 / 2.0

    def update_imu_data(self, data):
        self.roll_dot = data.qvel[3]
        self.pitch_dot = data.qvel[4]
        self.yaw_dot = data.qvel[5]
        quat = data.qpos[3:7]
        rpy = euler.quat2euler(quat)
        self.roll = rpy[0]
        self.pitch = rpy[1]
        self.yaw = rpy[2]

    # --- 以下是你的核心控制计算函数 (保持不变) ---
    def get_command_pitch(self, data):
    # 1. 获取基础速度环计算的目标角
        pitch_ref_from_vel = self.get_pitch_ref_from_velocity(data)

        # 2. 【核心注入】目标角 = 基础角 + RL 残差偏置
        pitch_d = pitch_ref_from_vel + self.rl_pitch_offset*1

        self.theta_pitch = pitch_d - self.pitch
        self.theta_pitch_pass = self.ST_SIT2_FLC_FM(self.theta_pitch)
        # 3. 直接通过纯 PID 计算力矩，不再经过模糊函数
        command = self.command_pitch.compute(error=self.theta_pitch, dt=self.model.opt.timestep)
        return command

    def get_pitch_ref_from_velocity(self, data):
        theta_velocity = self.velocity_d - self.robot_x_velocity    
        # print(f"robot_velocity:{self.robot_x_velocity}")
        pitch_offset = self.command_velocity.compute(error=theta_velocity, dt=self.model.opt.timestep)
        
        pitch_offset = np.clip(pitch_offset,
        -20.0 * math.pi / 180,
         20.0 * math.pi / 180
        )
        command = pitch_offset
        # return pitch_offset
        return command
    

    # def get_command_yaw(self, data):
    #     theta_yaw = self.yaw_d - self.yaw 
    #     return self.command_yaw.compute(error=theta_yaw, dt=self.model.opt.timestep)
    # # 在 get_command_yaw 中增加输出限幅

    def get_command_yaw(self, data):
        # 1. 角度归一化 (保持你之前的修改，非常好)
        raw_error = self.yaw_d - self.yaw
        theta_yaw = (raw_error + np.pi) % (2 * np.pi) - np.pi
        
        # 2. 计算 PID
        output = self.command_yaw.compute(error=theta_yaw, dt=self.model.opt.timestep)
        
        # 3. [新增] 输出限幅！
        # 转向力矩不能太大，否则会干扰平衡 (balance函数里它是直接加减的)
        # 假设最大平衡力矩是 5Nm，那转向最好不要超过 1.5Nm
        MAX_YAW_TORQUE = 3
        output = np.clip(output, -MAX_YAW_TORQUE, MAX_YAW_TORQUE)
        return output
    
    def com(self,xita_r1, xita_r2, xita_l1, xita_l2, roll):
        xita_r1=xita_r1-20*math.pi/180
        xita_r2=xita_r2-95*math.pi/180
        xita_l1=-xita_l1-20*math.pi/180
        xita_l2=-xita_l2-95*math.pi/180
        lr1 = 150 * 1e-3
        lr2 = 250 * 1e-3
        ll1 = lr1
        ll2 = lr2
        dlr = 355 * 1e-3
        # 重心位置
        dr1 = np.array([0.020031, 0.0032153, 0.0036019])
        dr2 = np.array([0.072881, 0.034135, 0.019602])
        dl1 = np.array([0.019888, 0.0032269, -0.0036874])
        dl2 = np.array([0.073104, 0.034096, -0.019634])
        db = np.array([0.01131, -0.0013618, -0.00028518])
        # 质量
        mr1 = 0.51378
        mr2 = 0.27188
        ml1 = 0.51183
        ml2 = 0.27057
        mb = 2.9637
        # 简便写法
        cr1 = np.cos(xita_r1)
        sr1 = np.sin(xita_r1)
        cl1 = np.cos(xita_l1)
        sl1 = np.sin(xita_l1)
        cr1r2 = np.cos(xita_r1 + xita_r2)
        sr1r2 = np.sin(xita_r1 + xita_r2)
        cl1l2 = np.cos(xita_l1 + xita_l2)
        sl1l2 = np.sin(xita_l1 + xita_l2)

        Xr1 = np.array([cr1 * dr1[0] - sr1 * dr1[1], sr1 * dr1[0] + cr1 * dr1[1], dr1[2] + dlr / 2])
        Xr2 = np.array([cr1r2 * dr2[0] - sr1r2 * dr2[1] + lr1 * cr1, sr1r2 * dr2[0] + cr1r2 * dr2[1] + lr1 * sr1, dr2[2] + dlr / 2])
        Xl1 = np.array([cl1 * dl1[0] - sl1 * dl1[1], sl1 * dl1[0] + cl1 * dl1[1], dl1[2] - dlr / 2])
        Xl2 = np.array([cl1l2 * dl2[0] - sl1l2 * dl2[1] + ll1 * cl1, sl1l2 * dl2[0] + cl1l2 * dl2[1] + ll1 * sl1, dl2[2] - dlr / 2])
        Xb = np.array([db[0], db[1], db[2]])
        

        T_80_3 = np.array([lr2 * np.cos(xita_r1 + xita_r2) / 2 + lr2 * np.cos(xita_l1 + xita_l2) / 2 + lr1 * np.cos(xita_l1) / 2 + lr1 * np.cos(xita_r1) / 2, 
                        lr2 * np.sin(xita_l1 + xita_l2) / 2 + lr2 * np.sin(xita_r1 + xita_r2) / 2 + lr1 * np.sin(xita_l1) / 2 + lr1 * np.sin(xita_r1) / 2,
                        0])

        # 质心计算
        Xbody = (mr1 * Xr1 + mr2 * Xr2 + ml1 * Xl1 + ml2 * Xl2 + mb * Xb) / (mr1 + mr2 + ml1 + ml2 + mb)
        X8_re = Xbody - T_80_3


        # 考虑机体 roll 的影响
        Rx_roll = np.array([[1, 0, 0], [0, np.cos(-roll), -np.sin(-roll)], [0, np.sin(-roll), np.cos(-roll)]])
        X8_re = np.dot(Rx_roll, X8_re)


        phi = np.arctan(X8_re[0] / X8_re[1])
        pitch_com = phi
        roll_com = np.arctan(X8_re[2] / X8_re[1]) 
        L = np.sqrt(X8_re[0] ** 2 + X8_re[1] ** 2 )
        # print("L",L)
        
        # 通过等效转动惯量来计算等效质量
        rr1 = np.sqrt((Xr1[0] - T_80_3[0]) ** 2 + (Xr1[1] - T_80_3[1]) ** 2)  # 计算Xr1相对于重心的距离
        rr2 = np.sqrt((Xr2[0] - T_80_3[0]) ** 2 + (Xr2[1] - T_80_3[1]) ** 2)  # 计算Xr2相对于重心的距离
        rl1 = np.sqrt((Xl1[0] - T_80_3[0]) ** 2 + (Xl1[1] - T_80_3[1]) ** 2)  # 计算Xl1相对于重心的距离
        rl2 = np.sqrt((Xl2[0] - T_80_3[0]) ** 2 + (Xl2[1] - T_80_3[1]) ** 2)  # 计算Xl2相对于重心的距离
        rb = np.sqrt((Xb[0] - T_80_3[0]) ** 2 + (Xb[1] - T_80_3[1]) ** 2)     # 计算Xb相对于重心的距离
        
        m = (mr2 * rr2 ** 2 + ml2 * rl2 ** 2 + mr1 * rr1 ** 2 + ml1 * rl1 ** 2 + mb * rb ** 2) / (L ** 2)
        J_z=m*L**2
        
        # 计算等效转动惯量(roll)
        rr1_r = math.sqrt((Xr1[0] - T_80_3[0])**2 + (Xr1[2] - T_80_3[2])**2)
        rr2_r = math.sqrt((Xr2[0] - T_80_3[0])**2 + (Xr2[2] - T_80_3[2])**2)
        rl1_r = math.sqrt((Xl1[0] - T_80_3[0])**2 + (Xl1[2] - T_80_3[2])**2)
        rl2_r = math.sqrt((Xl2[0] - T_80_3[0])**2 + (Xl2[2] - T_80_3[2])**2)
        rb_r = math.sqrt((Xb[0] - T_80_3[0])**2 + (Xb[2] - T_80_3[2])**2)
        
        J_y = mr2 * rr2_r**2 + ml2 * rl2_r**2 + mr1 * rr1_r**2 + ml1 * rl1_r**2 + mb * rb_r**2
        # print("L, m, pitch_com, roll_com,J_z,J_y",L, m, pitch_com, roll_com,J_z,J_y)
        return L, m, pitch_com, roll_com,J_z,J_y
     

    def ST_SIT2_FLC_FM(self, error_pass):
        """
        重构版二型模糊控制器：彻底解决量纲风险与边界跳变
        """
        # =========================================================
        # 1. 论域归一化 (关键：将论域放宽到 35 度，防止爬坡时频繁溢出)
        # =========================================================
        DOMAIN_DEG = 15.0
        # 将输入弧度 error_pass 转换为归一化比例 [-1, 1]
        error = error_pass / math.pi * 180.0 / DOMAIN_DEG
        
        # 定义论域对应的最大弧度，用于后续统一量纲
        MAX_RAD = DOMAIN_DEG / 180.0 * math.pi

        # 提取参数
        gamma_d_g = self.fuzzy_params["gamma_d_g"]
        eth = self.fuzzy_params["eth"]
        a = self.fuzzy_params["a"]

        # =========================================================
        # 2. 动态数学保护屏障 (防止 ld_g 出现负数或数值爆炸)
        # =========================================================
        cos_gamma = np.cos(gamma_d_g)
        max_safe_eth = cos_gamma - 0.001
        eth = np.minimum(eth, max_safe_eth)

        # ---------- 基础中间变量计算 ----------
        ld_g = (cos_gamma - eth) / cos_gamma
        B0_g, B1_g, B2_g = 0.0, ld_g * np.sin(gamma_d_g), 1.0
        C0_g, C1_g, C2_g = 0.0, ld_g * np.cos(gamma_d_g), 1.0
        
        K0_g = (B1_g - B0_g) / (C1_g - C0_g)
        K1_g = (B2_g - B1_g) / (C2_g - C1_g)
        N0_g = (B0_g * C1_g - B1_g * C0_g) / (C1_g - C0_g)
        N1_g = (B1_g * C2_g - B2_g * C1_g) / (C2_g - C1_g)
        
        ld_max, ld_min = 0.9, 0.3

        # ---------- gamma_g 分段计算 ----------
        if error <= -C2_g:
            gamma_g = B2_g * 70.0 + 10.0
        elif -C2_g < error < -C1_g:
            gamma_g = (K1_g * abs(error) + N1_g) * 70.0 + 10.0
        elif -C1_g <= error < C1_g:
            gamma_g = (K0_g * abs(error)) * 70.0 + 10.0
        elif C1_g <= error < C2_g:
            gamma_g = (K1_g * error + N1_g) * 70.0 + 10.0
        else:
            gamma_g = B2_g * 70.0 + 10.0

        # ---------- ld_e 与 比例系数计算 ----------
        gamma_g_1 = gamma_g * np.pi / 180.0
        ld_e = abs(error) / np.cos(gamma_g_1)
        ld_e = max(ld_min, min(ld_e, ld_max))
        ld = ld_e

        B0, B1, B2 = 0.0, ld * np.sin(gamma_g_1), 1.0
        C0, C1, C2 = 0.0, ld * np.cos(gamma_g_1), 1.0
        m0, m1, m2 = a, 1.0 - a, a

        # 避免除以 0 的极小偏置
        eps = 1e-8
        K0 = 0.5 * ((B1 - B0 * m0) / (C1 * m0 - C0 + abs(error) * (-m0 + 1.0) + eps) +
                    (B0 - B1 * m1) / (C0 * m1 - C1 + abs(error) * (-m1 + 1.0) + eps))
        K1 = 0.5 * ((B2 - B1 * m1) / (C2 * m1 - C1 + abs(error) * (-m1 + 1.0) + eps) +
                    (B1 - B2 * m2) / (C1 * m2 - C2 + abs(error) * (-m2 + 1.0) + eps))
        N0 = 0.5 * ((B1 * C0 - B0 * C1 * m0) / (-C1 * m0 + C0 + abs(error) * (m0 - 1.0) + eps) +
                    (B0 * C1 - B1 * C0 * m1) / (-C0 * m1 + C1 + abs(error) * (m1 - 1.0) + eps))
        N1 = 0.5 * ((B2 * C1 - B1 * C2 * m1) / (-C2 * m1 + C1 + abs(error) * (m1 - 1.0) + eps) +
                    (B1 * C2 - B2 * C1 * m2) / (-C1 * m2 + C2 + abs(error) * (m2 - 1.0) + eps))

        # =========================================================
        # 3. φ 分段计算 (核心修复：统一使用 MAX_RAD 保证量纲正确)
        # =========================================================
        # 逻辑：每一段的输出都由 [比例值 * 最大弧度] 组成，确保 phi 的单位是弧度
        if error <= -C2:
            # 修复：边界外推使用边界斜率 K1，且量纲对齐
            phi = (-B2 * MAX_RAD) + (error + C2) * (K1 * MAX_RAD)
        elif -C2 < error < -C1:
            phi = (K1 * error - N1) * MAX_RAD
        elif -C1 <= error < C1:
            phi = (K0 * error) * MAX_RAD
        elif C1 <= error < C2:
            phi = (K1 * error + N1) * MAX_RAD
        else: 
            # 修复：边界外推量纲对齐
            phi = (B2 * MAX_RAD) + (error - C2) * (K1 * MAX_RAD)
        
        # 4. 返回最终补偿弧度 (如果你手动测试觉得增益不够，可以把 1.0 改大)
        scaling_factor = self.fuzzy_params["k_out"]
        return phi * scaling_factor
    
    # [新增] 重置接口
    def reset(self):
        """当环境重置时调用，清除 PID 记忆"""
        self.command_yaw.reset()
        self.command_pitch.reset()
        self.command_velocity.reset()
        self.velocity_d = 0.0
        self.yaw_d = 0.0
        print(">>> LQR 控制器已重置 (PID 记忆清除)")

    def balance(self, data):
        a = self.get_command_pitch(data) # 串级
        b = self.get_pitch_ref_from_velocity(data)
        c = self.get_command_yaw(data) # 并级
        left_torque = -0.5 * a - 0.5 * b - c
        right_torque = 0.5 * a + 0.5 * b - c
        data.ctrl[self.actuator_ids["left_wheel"]] = left_torque
        data.ctrl[self.actuator_ids["right_wheel"]] = right_torque

        

# --- VMC 控制器 (结构已优化) ---
class VMC:
    def __init__(self, model):
        # 修改 1: 初始化不再传入 data，只传入 model
        self.model = model
        
        # 动态获取腿部执行器 ID
        self.act_ids = {
            "L1": model.actuator("L1_joint_ctrl").id,
            "L2": model.actuator("L2_joint_ctrl").id,
            "R1": model.actuator("R1_joint_ctrl").id,
            "R2": model.actuator("R2_joint_ctrl").id
        }
        
        # 预计算一些关节ID以便在update中使用
        self.joint_handles = {}
        for name in ["L1", "L2", "R1", "R2"]:
             j_id = model.joint(f"{name}_joint").id
             self.joint_handles[name] = {
                 "id": j_id,
                 "qpos": model.jnt_qposadr[j_id],
                 "dof": model.jnt_dofadr[j_id]
             }

        # 状态变量
        self.roll, self.roll_dot = 0.0, 0.0
        self.q = {"L1": 0.0, "L2": 0.0, "R1": 0.0, "R2": 0.0}
        self.v = {"L1": 0.0, "L2": 0.0, "R1": 0.0, "R2": 0.0}

    def update_states(self, data):
        """
        修改 2: 显式的状态更新函数，与 LQR 结构对齐
        """
        # 1. 更新 IMU 信息
        quat = data.qpos[3:7]
        self.roll = euler.quat2euler(quat)[0]
        self.roll_dot = data.qvel[3]

        # 2. 更新关节信息
        for name, handle in self.joint_handles.items():
            self.q[name] = data.qpos[handle["qpos"]]
            self.v[name] = data.qvel[handle["dof"]]

    def vmc(self, data):
        
        # 参数
        k, d = 300, 15
        theta_10, theta_20 = 70 * math.pi / 180, 95 * math.pi / 180
        l1, l2 = 0.15, 0.25
        x_d, z_d = 0.0353, 0.32
        
        # 补偿
        z_d_comp = 2500 * self.roll + 100 * self.roll_dot
        # gravity_comp = 22.0

        # --- 左腿 ---
        q_l1, q_l2 = self.q["L1"], self.q["L2"]
        v_l1, v_l2 = self.v["L1"], self.v["L2"]
        
        # 正运动学
        x_l = l1 * math.sin(theta_10 - q_l1) - l2 * math.sin((theta_20 + q_l2) - (theta_10 - q_l1))
        z_l = l1 * math.cos(theta_10 - q_l1) + l2 * math.cos((theta_20 + q_l2) - (theta_10 - q_l1))
        
        # 雅可比
        J_l = np.array([
            [-l1*math.cos(theta_10-q_l1)-l2*math.cos((theta_20+q_l2)-(theta_10-q_l1)), -l2*math.cos((theta_20+q_l2)-(theta_10-q_l1))],
            [ l1*math.sin(theta_10-q_l1)-l2*math.sin((theta_20+q_l2)-(theta_10-q_l1)), -l2*math.sin((theta_20+q_l2)-(theta_10-q_l1))]
        ])
        
        p_dot_l = J_l @ np.array([v_l1, v_l2])
        F_l = np.array([
            k * (x_d - x_l) - d * p_dot_l[0],
            k * (z_d - z_l) - d * p_dot_l[1] - z_d_comp #+ gravity_comp
        ])
        tau_l = J_l.T @ F_l

        # --- 右腿 ---
        q_r1, q_r2 = self.q["R1"], self.q["R2"]
        v_r1, v_r2 = self.v["R1"], self.v["R2"]

        x_r = l1 * math.sin(theta_10 + q_r1) - l2 * math.sin((theta_20 - q_r2) - (theta_10 + q_r1))
        z_r = l1 * math.cos(theta_10 + q_r1) + l2 * math.cos((theta_20 - q_r2) - (theta_10 + q_r1))
        
        J_r = np.array([
            [ l1*math.cos(theta_10+q_r1)+l2*math.cos((theta_20-q_r2)-(theta_10+q_r1)),  l2*math.cos((theta_20-q_r2)-(theta_10+q_r1))],
            [-l1*math.sin(theta_10+q_r1)+l2*math.sin((theta_20-q_r2)-(theta_10+q_r1)),  l2*math.sin((theta_20-q_r2)-(theta_10+q_r1))]
        ])

        p_dot_r = J_r @ np.array([v_r1, v_r2])
        F_r = np.array([
            k * (x_d - x_r) - d * p_dot_r[0],
            k * (z_d - z_r) - d * p_dot_r[1] + z_d_comp #+ gravity_comp
        ])
        tau_r = J_r.T @ F_r

        # --- 下发指令 ---
        data.ctrl[self.act_ids["L1"]] = tau_l[0]
        data.ctrl[self.act_ids["L2"]] = tau_l[1]
        data.ctrl[self.act_ids["R1"]] = tau_r[0]
        data.ctrl[self.act_ids["R2"]] = tau_r[1]

# --- Main 函数 ---
def main():
    
    from mj_keyboard import KeyboardCommander
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path("/home/huang/wheel_leg/wheel_legged_genesis_new/assets/description/urdf/scence.xml")
    data = mujoco.MjData(model)
    step = 0
    # 2. 实例化控制器
    # 修改 3: 这里的结构完全一致了，都不再传入 data
    vmc_ctrl = VMC(model)
    lqr_ctrl = LQR_Controller(model)    
    
    # 这里自定义灵敏度，例如速度每级 0.2，转向每级 5 度
    cmd = KeyboardCommander(v_step=0.1, yaw_step_deg=10.0)

    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     model.opt.timestep = 0.002
    #     print("仿真开始...")

    #     while viewer.is_running():


            
    #         step_start = time.time()
    #         step += 1
    #         if step % 100 == 0:
    #             print(f"pitch: {lqr_ctrl.pitch}, yaw: {lqr_ctrl.yaw}, roll: {lqr_ctrl.roll}")
    #             print(f"pitch_dot: {lqr_ctrl.pitch_dot}, velocity: {lqr_ctrl.robot_x_velocity}")
    #             print(f"vel_error: {lqr_ctrl.velocity_d - lqr_ctrl.robot_x_velocity}")
    #             print("====================================================================")
    #         # 获取最新指令，无需关心多线程逻辑
    #         target_v, target_yaw = cmd.get_command()

    #         # 注入指令到 LQR 控制器 ---
    #         lqr_ctrl.velocity_d = target_v
    #         lqr_ctrl.yaw_d = target_yaw

    #         # --- A. 统一更新状态 ---
    #         # 两个控制器都先去读取它们需要的数据
    #         lqr_ctrl.update_imu_data(data)
    #         lqr_ctrl.update_joint_states(data)
            
    #         # 修改 4: VMC 也在这里显式更新，而不是在 mc 函数内部隐式处理
    #         # 这样保证了 vmc 函数只负责“计算”，update 只负责“读取”，逻辑解耦
    #         vmc_ctrl.update_states(data)

            
    #         # --- B. 统一计算控制 ---
    #         # 修改 5: 调用 mc 计算腿部力矩
    #         vmc_ctrl.vmc(data)     
            
    #         # 调用 balance 计算轮子力矩
    #         lqr_ctrl.balance(data) 

            
    #         # --- C. 物理步进 ---
    #         mujoco.mj_step(model, data)
    #         viewer.sync()

    #         # 频率控制
    #         elapsed = time.time() - step_start
    #         if elapsed < model.opt.timestep:
    #             time.sleep(model.opt.timestep - elapsed)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        model.opt.timestep = 0.002
        
        # --- 1. 设置渲染解频参数 ---
        render_fps = 60  # 目标画面刷新率 (60Hz 足够流畅)
        # 计算每渲染一帧，需要跑多少次物理步进 (1/60 ÷ 0.002 ≈ 8次)
        physics_steps_per_render = int((1.0 / render_fps) / model.opt.timestep) 
        
        print("仿真开始...")

        while viewer.is_running():
            # 使用更精确的性能计数器，替代 time.time()
            step_start = time.perf_counter()

            # --- 2. 物理与控制高频循环 (500Hz) ---
            for _ in range(physics_steps_per_render):
                step += 1
                
                # 获取指令
                target_v, target_yaw = cmd.get_command()
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
                
                # (可选) 控制台打印频率调整，避免刷屏拖慢速度
                if step % 500 == 0:  # 每 1 秒打印一次
                    print(f"pitch: {lqr_ctrl.pitch:.3f}, yaw: {lqr_ctrl.yaw:.3f}, roll: {lqr_ctrl.roll:.3f}")
                    print(f"pitch_dot: {lqr_ctrl.pitch_dot:.3f}, velocity: {lqr_ctrl.robot_x_velocity:.3f}")
                    print(f"vel_error: {lqr_ctrl.velocity_d - lqr_ctrl.robot_x_velocity:.3f}")
                    print("-" * 50)

            # --- 3. 画面渲染低频循环 (60Hz) ---
            viewer.sync()

            # --- 4. 挂钟时间真实同步 ---
            # 计算这批物理仿真理论上需要消耗的总时间 (例如 8 * 0.002 = 0.016s)
            expected_time = physics_steps_per_render * model.opt.timestep
            elapsed = time.perf_counter() - step_start
            
            # 如果计算得比现实时间快，就休眠补齐差价，实现 1:1 播放
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)


if __name__ == "__main__":
    main()