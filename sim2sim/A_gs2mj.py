import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R

# =============================================================================
# 核心类：MuJoCo 原生控制器 (完全脱离 Genesis)
# =============================================================================
class NativeController:
    def __init__(self, model, data):
        self.m = model
        self.d = data
        
        # ---------------------------------------------------------------------
        # 1. 物理参数配置 (需要与你的 URDF/Genesis 配置一致)
        # ---------------------------------------------------------------------
        # 连杆长度 (米) - 假设值，请根据你的 URDF 修改！
        # l1: 大腿长度 (Hip to Knee)
        # l2: 小腿长度 (Knee to Wheel)
        self.l1 = 0.15  # 示例值，请核对 URDF
        self.l2 = 0.25  # 示例值，请核对 URDF
        
        # VMC 延伸控制 (原有)
        self.vmc_kp = 400.0   
        self.vmc_kd = 40.0
        
        # 平衡控制参数 (参考你原来的 LQR/PID)
        self.pitch_kp = 20.0
        self.pitch_kd = 1.0
        self.vel_kp = 10.0
        self.vel_ki = 0.1
        self.vel_integral = 0.0 # 速度积分项
        
        # 目标状态
        self.target_height = 0.32
        self.target_vel = 0.0
        self.target_yaw_dot = 0.0

        # ---------------------------------------------------------------------
        # 2. 缓存关节索引 (避免在循环里查表，提高原生效率)
        # ---------------------------------------------------------------------
        # 必须与 XML 定义一致
        self.idx_map = {
            'R1': mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'R1_joint'),
            'R2': mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'R2_joint'),
            'R3': mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'R3_joint'),
            'L1': mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'L1_joint'),
            'L2': mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'L2_joint'),
            'L3': mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'L3_joint'),
        }
        
        # 获取 qpos 和 qvel 的地址
        self.qpos_adr = {k: self.m.jnt_qposadr[v] for k, v in self.idx_map.items()}
        self.dof_adr = {k: self.m.jnt_dofadr[v] for k, v in self.idx_map.items()}
        
        # 获取 Actuator 地址 (XML actuator 顺序: R1, R2, R3, L1, L2, L3)
        # 这种硬编码顺序最稳，对应 d.ctrl 的索引
        self.act_idx = {
            'R1': 0, 'R2': 1, 'R3': 2,
            'L1': 3, 'L2': 4, 'L3': 5
        }

    def get_state(self):
        """原生获取 MuJoCo 状态"""
        # 1. 基座姿态 (四元数 [w,x,y,z])
        # MuJoCo qpos[3:7] 是四元数
        quat_mj = self.d.qpos[3:7] 
        # Scipy 需要 [x,y,z,w]
        quat_scipy = [quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]]
        r = R.from_quat(quat_scipy)
        euler = r.as_euler('xyz', degrees=False)
        self.roll, self.pitch, self.yaw = euler[0], euler[1], euler[2]
        
        # 2. 基座角速度 (Body Frame)
        self.omega_body = self.d.qvel[3:6] # MuJoCo freejoint qvel 角速度部分是在 Body Frame 下的
        self.pitch_dot = self.omega_body[1]
        self.yaw_dot = self.omega_body[2]

        # 3. 基座线速度 (需转换到 Body Frame)
        # qvel[0:3] 是 World Frame 下的速度 (如果使用 freejoint)
        vel_world = self.d.qvel[0:3]
        # 使用旋转矩阵转到 Body Frame
        rot_mat = r.as_matrix() # Body to World
        self.vel_body = rot_mat.T @ vel_world # World to Body
        self.vx = self.vel_body[0]

        # 4. 关节状态
        def q(name): return self.d.qpos[self.qpos_adr[name]]
        def v(name): return self.d.qvel[self.dof_adr[name]]
        
        self.q_left = {'hip': q('L1'), 'knee': q('L2'), 'wheel': q('L3')}
        self.q_right = {'hip': q('R1'), 'knee': q('R2'), 'wheel': q('R3')}
        
        self.dq_left = {'hip': v('L1'), 'knee': v('L2'), 'wheel': v('L3')}
        self.dq_right = {'hip': v('R1'), 'knee': v('R2'), 'wheel': v('R3')}

    def compute_leg_kinematics(self, hip_angle, knee_angle):
        """
        计算单腿正运动学 (Forward Kinematics) 和 雅可比 (Jacobian)
        返回: 
            L: 虚拟腿长 (Hip 到 Wheel Center)
            J: 雅可比矩阵 (2x2) -> [dL/dHip, dL/dKnee] (简化版，只关注长度变化)
        """
        # 这里使用余弦定理或几何投影计算虚拟腿长 L
        # 假设关节 0 度是自然下垂，hip 向前为正，knee 向后为正 (根据具体 URDF 定义修改)
        # 简化的二连杆模型：
        # 设 alpha = hip_angle, beta = knee_angle
        # 你的机器人结构通常是：Hip关节转动大腿，Knee关节转动小腿
        # 虚拟腿长 L^2 = l1^2 + l2^2 - 2*l1*l2*cos(pi - beta) 
        # (具体公式取决于你的零位定义，这里使用通用的二连杆几何)
        
        # 假设：膝关节 theta2 是相对大腿的角度。
        # 腿长 L 计算
        # 注意：需要根据你的 URDF 零位校准公式。
        # 假设伸直时 knee=0 -> L = l1 + l2
        # 或者弯曲 90 度时 knee=pi/2
        
        # 简单的三角函数解算 (Standard VMC):
        # 设 theta_knee 是膝关节夹角
        L = np.sqrt(self.l1**2 + self.l2**2 + 2 * self.l1 * self.l2 * np.cos(knee_angle))
        
        # 雅可比 (力到力矩的映射)
        # 我们只做长度控制 (Spring Force along L)，所以需要 dL/d_q
        # dL/d_knee = - (l1*l2*sin(knee)) / L
        # dL/d_hip = 0 (假设 hip 只改变腿的指向，不改变长度，VMC模型中通常这样解耦)
        
        # 防除零
        if L < 0.01: L = 0.01
        
        dL_dknee = -(self.l1 * self.l2 * np.sin(knee_angle)) / L
        dL_dhip = 0.0 
        
        # 这是一个简化的 VMC Jacobian，仅映射垂直力 F_radial
        J_T = np.array([dL_dhip, dL_dknee]) 
        
        return L, J_T

    def run_vmc(self, side='left'):
        """计算单腿 VMC 力矩"""
        if side == 'left':
            q = self.q_left
            dq = self.dq_left
        else:
            q = self.q_right
            dq = self.dq_right
            
        # 1. 计算当前腿长
        current_L, J_T = self.compute_leg_kinematics(q['hip'], q['knee'])
        
        # 2. 也是计算腿长变化率 dL (直接用雅可比乘关节速度)
        dL = J_T[0] * dq['hip'] + J_T[1] * dq['knee']
        
        # 3. 弹簧阻尼力 (F = K(L_des - L) - D * dL)
        # 目标是支撑基座高度
        # 简单近似：L_des = target_height / cos(roll) / cos(pitch)
        L_des = self.target_height # 简化，假设身体平直
        
        F_radial = self.vmc_kp * (L_des - current_L) - self.vmc_kd * dL
        
        # 限制力的大小 (模拟电机扭矩限制)
        F_radial = np.clip(F_radial, -500, 500)
        
        # 4. 映射回关节力矩 (tau = J^T * F)
        # 注意方向！如果算出的是推力，需要确认电机正方向
        tau_hip = J_T[0] * F_radial
        tau_knee = J_T[1] * F_radial
        
        # 【重要】重力补偿 (Gravity Compensation)
        # 简单的重力前馈，支撑机身重量
        # F_ff = m * g / 2 (两条腿)
        # F_total = F_radial + F_ff
        
        return tau_hip, tau_knee

    def run_balance(self):
        """计算平衡控制力矩 (LQR/PID)"""
        # 1. 误差计算
        pitch_err = self.pitch - 0.0 # 目标 pitch = 0
        vel_err = self.vx - self.target_vel
        
        # 2. 积分项 (已修复 NameError 和 SyntaxError)
        # 自动从模型中获取 dt (通常是 0.002)
        dt = self.m.opt.timestep 
        self.vel_integral += vel_err * dt
        
        # 抗积分饱和
        self.vel_integral = np.clip(self.vel_integral, -1.0, 1.0) 
        
        # 3. 计算力矩 (Balance Torque)
        # T = kp*theta + kd*dtheta + kv*v + ki*iv
        balance_torque = (self.pitch_kp * pitch_err + 
                          self.pitch_kd * self.pitch_dot +
                          self.vel_kp * vel_err + 
                          self.vel_ki * self.vel_integral)
        
        # 转向控制 (Yaw)
        yaw_torque = 1.0 * (self.yaw_dot - self.target_yaw_dot)
        
        # 分配给左右轮
        # 平衡力矩是同向的，转向力矩是差动的
        tau_left_wheel = balance_torque - yaw_torque
        tau_right_wheel = balance_torque + yaw_torque
        
        return tau_left_wheel, tau_right_wheel

    def update(self):
        """主控制循环：读取状态 -> 计算 -> 输出"""
        # 1. 获取最新状态
        self.get_state()
        
        # 2. 计算 VMC (腿部)
        l_hip_tau, l_knee_tau = self.run_vmc('left')
        r_hip_tau, r_knee_tau = self.run_vmc('right')
        
        # 3. 计算 Balance (轮子)
        l_wheel_tau, r_wheel_tau = self.run_balance()
        
        # 4. 写入 MuJoCo (Actuation)
        # 必须严格对应 XML Actuator 顺序
        # 再次确认你的 XML 顺序: R1, R2, R3, L1, L2, L3
        
        # 右腿
        self.d.ctrl[self.act_idx['R1']] = -r_hip_tau    # 注意方向！根据之前的测试加负号
        self.d.ctrl[self.act_idx['R2']] = -r_knee_tau   # 注意方向！
        self.d.ctrl[self.act_idx['R3']] = r_wheel_tau
        
        # 左腿
        self.d.ctrl[self.act_idx['L1']] = -l_hip_tau
        self.d.ctrl[self.act_idx['L2']] = -l_knee_tau
        self.d.ctrl[self.act_idx['L3']] = l_wheel_tau

        return [l_hip_tau, l_wheel_tau] # 仅用于调试打印

# =============================================================================
# 主仿真循环
# =============================================================================
def main():
    xml_path = "sim2sim/scence.xml" # 确保路径正确
    if not os.path.exists(xml_path):
        print("❌ 找不到 scence.xml")
        return

    print(f"Loading: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # 实例化原生控制器
    controller = NativeController(m, d)
    
    print("🚀 开始原生 MuJoCo 仿真...")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 初始状态设置
        d.qpos[2] = 0.4 # 抬高一点
        
        # 仿真参数
        dt = m.opt.timestep
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- 控制循环 ---
            debug_info = controller.update()
            
            # --- 物理步进 ---
            mujoco.mj_step(m, d)
            viewer.sync()
            
            # 打印调试信息 (每 0.5s)
            if int(time.time()*10) % 5 == 0:
                print(f"\rRoll:{controller.roll:.3f} | Pitch:{controller.pitch:.3f} | L_Tau:{debug_info[0]:.2f}", end="")
            
            # 实时性控制
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()