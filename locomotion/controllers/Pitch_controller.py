import math
from transforms3d import euler
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat  # type: ignore
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import torch
import time

import torch

class BatchedPIDController:
    def __init__(self, kp, ki, kd, num_envs, device=None, dtype=torch.float32):
        self.kp = torch.as_tensor(kp, dtype=dtype, device=device)
        self.ki = torch.as_tensor(ki, dtype=dtype, device=device)
        self.kd = torch.as_tensor(kd, dtype=dtype, device=device)

        self.device = device if device is not None else torch.device("cpu")

        # (num_envs,)
        self.prev_error = torch.zeros(num_envs, dtype=dtype, device=self.device)
        self.integral   = torch.zeros(num_envs, dtype=dtype, device=self.device)

    def compute(self, error, dt, stopping_mask=None):
        # 1. 默认状态下保持较弱的衰减，维持爬坡动力
        leak_rate = 0.999 
        
        # 2. [核心修改] 当处于静止掩码时，使用极强的衰减 (0.5)
        # 这样积分不会瞬间消失(导致剧烈跳动)，但在 2-3 帧内会迅速归零
        if stopping_mask is not None:
            # 在需要定住的环境，让积分迅速缩水
            leak_rate_tensor = torch.where(stopping_mask, 1, 1)
            self.integral *= leak_rate_tensor
        else:
            self.integral *= leak_rate

        self.integral += error * dt
        # 增大限幅，给爬坡留足力矩空间
        self.integral = torch.clamp(self.integral, -100.0, 100.0) 

        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error  
        return output



class BatchedLQRController:
    """
    将你原来的 LQR_Controller 初始化部分向量化（batched，num_envs）
    使用示例：
        ctrl = BatchedLQRController(num_envs, robot, device=device)
    假定 robot 在并行模式下的接口返回形状 (num_envs, ...) 的 torch tensors
    """
    def __init__(self, num_envs: int, device: torch.device = None):
        self.num_envs = int(num_envs)
        self.device = torch.device('cpu') if device is None else device
        # self.robot = robot
        self.dt = 0.002
        self.joint_name_to_idx = None





        # base_quat 保留原意，但并行化为 tensor (num_envs, 4)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        # ================== 状态量（全部张量化，保证每个 env 独立） ==================
        z = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        # position / velocity / efforts for relevant joints
        self.left_wheel_velocity  = z.clone()
        self.left_wheel_position  = z.clone()
        self.right_wheel_velocity = z.clone()
        self.right_wheel_position = z.clone()
        self.left_hip_position    = z.clone()
        self.right_hip_position   = z.clone()
        self.left_knee_position   = z.clone()
        self.right_knee_position  = z.clone()

        self.left_hip_effort  = z.clone()
        self.left_knee_effort = z.clone()
        self.left_wheel_effort = z.clone()
        self.right_hip_effort = z.clone()
        self.right_knee_effort = z.clone()
        self.right_wheel_effort = z.clone()

        # imu / com related
        self.roll       = z.clone()
        self.pitch      = z.clone()
        self.pitch_com  = z.clone()
        self.roll_com   = z.clone()
        self.n          = z.clone()
        self.pitch_dot  = z.clone()
        self.left_knee_velocity  = z.clone()
        self.right_knee_velocity = z.clone()
        self.left_hip_velocity   = z.clone()
        self.right_hip_velocity  = z.clone()

        # dynamics intermediate vars (batched)
        self.theta_r1 = z.clone(); self.theta_r2 = z.clone(); self.theta_r3 = z.clone()
        self.theta_rb = z.clone()
        self.theta_r0_dot = z.clone(); self.theta_r1_dot = z.clone(); self.theta_r2_dot = z.clone()
        self.theta_r3_dot = z.clone(); self.theta_rb_dot = z.clone()

        self.theta_l1 = z.clone(); self.theta_l2 = z.clone(); self.theta_l3 = z.clone()
        self.theta_lb = z.clone()
        self.theta_l0_dot = z.clone(); self.theta_l1_dot = z.clone(); self.theta_l2_dot = z.clone()
        self.theta_l3_dot = z.clone(); self.theta_lb_dot = z.clone()

        # torques / control outputs
        self.tao_rh = z.clone(); self.tao_rk = z.clone(); self.tao_rw = z.clone()
        self.tao_lh = z.clone(); self.tao_lk = z.clone(); self.tao_lw = z.clone()

        # other scalar states
        self.error_pitch = z.clone()
        self.error_pitch_dot = z.clone()
        self.desired_state = z.clone()
        self.prev_error = z.clone()
        self.integral = z.clone()
        self.robot_x_velocity = z.clone()
        self.integral_yaw = z.clone()
        self.yaw_dot = z.clone()
        self.yaw = z.clone()
        self.robot_x_position = z.clone()
        self.velocity_d = z.clone()
        self.roll_dot = z.clone()


        self.pitch_d = torch.zeros(num_envs, device=self.device) # 特殊变量保留
        self.velocity_d = torch.zeros(num_envs, device=self.device)
        self.yaw_d = torch.zeros(num_envs, device=self.device)
        self.yaw_d_dot = torch.zeros(num_envs, device=self.device)
        self.theta_pitch = torch.zeros(num_envs, device=self.device)



        # PID 控制器结构初始化 (参数稍后填充)
        # self.command_yaw = BatchedPIDController(kp=0.5, ki=0.001, kd=0, num_envs=self.num_envs, device=self.device)
        # # 最优的角度 PID 参数 (2026-02-22)：kp=175, ki=0.3, kd=1.3
        # self.command_pitch = BatchedPIDController(kp=175, ki=0.3, kd=1.3, num_envs=self.num_envs, device=self.device)
        # # 最优的速度 PID 参数 (2026-02-22)：kp=0.5, ki=0.05, kd=0
        # self.command_velocity = BatchedPIDController(kp=0.5, ki=0.05, kd=0, num_envs=self.num_envs, device=self.device)
        # #无训练时，走wave地形的最大速度误差为0.29，待对比 
        # 在 Pitch_controller.py 的 __init__ 中修改：
        self.command_yaw = BatchedPIDController(kp=1.0, ki=0.01, kd=0.2, num_envs=self.num_envs, device=self.device)
        self.command_pitch = BatchedPIDController(kp=30, ki=0.0, kd=0.5, num_envs=self.num_envs, device=self.device)
        self.command_velocity = BatchedPIDController(kp=0.8, ki=0.5, kd=0.0, num_envs=self.num_envs, device=self.device)

        self.params = {}   # 存放实时参数 Tensor
        self.defaults = {} # [新增] 存放默认值 float (用于 Reset)


    def set_joint_mapping(self, joint_name_to_idx):
        """设置关节索引映射（由环境传入）"""
        self.joint_name_to_idx = joint_name_to_idx



    def set_commands(self, velocity_d=None, yaw_d=None):
        """支持外界指令设置，必须为 shape (num_envs,) 的张量"""
        if velocity_d is not None:
            self.velocity_d = velocity_d
        if yaw_d is not None:
            self.yaw_d = yaw_d


    
    def set_state(self, dof_pos, dof_vel, base_vel, base_euler, base_ang ):
        """
        
        Args:
            dof_pos: [num_envs, num_dofs] 所有关节位置
            dof_vel: [num_envs, num_dofs] 所有关节速度
            quat: [num_envs, 4] 机体四元数
            ang: [num_envs, 3] 机体角速度
        """
        # 确保数据在正确设备上
        dof_pos = dof_pos.to(self.device)
        dof_vel = dof_vel.to(self.device)
        base_vel = base_vel.to(self.device)
        base_euler = base_euler.to(self.device)

        # quat = quat.to(self.device)
        ang = base_ang.to(self.device)
        
        # 提取需要的关节
        self.position_lw = dof_pos[:, self.joint_name_to_idx["left_wheel"]]
        self.position_rw = dof_pos[:, self.joint_name_to_idx["right_wheel"]]
        self.position_lh = dof_pos[:, self.joint_name_to_idx["left_hip"]]
        self.position_rh = dof_pos[:, self.joint_name_to_idx["right_hip"]]
        self.position_lk = dof_pos[:, self.joint_name_to_idx["left_knee"]]
        self.position_rk = dof_pos[:, self.joint_name_to_idx["right_knee"]]
        
        self.velocity_lw = dof_vel[:, self.joint_name_to_idx["left_wheel"]]
        self.velocity_rw = dof_vel[:, self.joint_name_to_idx["right_wheel"]]
        self.velocity_lh = dof_vel[:, self.joint_name_to_idx["left_hip"]]
        self.velocity_rh = dof_vel[:, self.joint_name_to_idx["right_hip"]]
        self.velocity_lk = dof_vel[:, self.joint_name_to_idx["left_knee"]]
        self.velocity_rk = dof_vel[:, self.joint_name_to_idx["right_knee"]]
        
        # 更新 IMU 数据
        self.roll_dot = ang[:, 0] #*torch.pi/180
        self.pitch_dot = ang[:, 1] #*torch.pi/180
        self.yaw_dot = ang[:, 2]    #*torch.pi/180
        
        # self.base_quat = quat
        # self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        # aligned_quat = transform_quat_by_quat(
        #     torch.ones_like(self.base_quat) * self.inv_base_init_quat,
        #     self.base_quat
        # )
        # rpy = quat_to_xyz(aligned_quat)
        euler = base_euler
        
        self.roll = euler[:, 0] *torch.pi/180
        self.pitch = euler[:, 1] *torch.pi/180
        self.yaw = euler[:, 2] *torch.pi/180

        # ====== 动力学模型参数计算(保留你原始公式) ======
        # pitch, pitch_dot 在 update_imu_data() 更新
        pitch     = self.pitch      # shape: [num_envs]
        pitch_dot = self.pitch_dot  # shape: [num_envs]
        # right leg
        self.theta_r1 = self.position_rw
        self.theta_r2 = -(95/180*math.pi - self.position_rk)
        self.theta_r3 = self.position_rh - 20/180*math.pi
        self.theta_rb = -self.theta_r2 - self.theta_r3 - math.pi/2 + pitch

        self.theta_r1_dot = self.velocity_rw
        self.theta_r2_dot = self.velocity_rk
        self.theta_r3_dot = self.velocity_rh
        self.theta_rb_dot = -self.theta_r2_dot - self.theta_r3_dot + pitch_dot

        # left leg
        self.theta_l1 = -self.position_lw
        self.theta_l2 = -(95/180*math.pi + self.position_lk)
        self.theta_l3 = -self.position_lh - 20/180*math.pi
        self.theta_lb = -self.theta_l2 - self.theta_l3 - math.pi/2 + pitch

        self.theta_l1_dot = -self.velocity_lw
        self.theta_l2_dot = -self.velocity_lk
        self.theta_l3_dot = -self.velocity_lh
        self.theta_lb_dot = -self.theta_l2_dot - self.theta_l3_dot + pitch_dot

        # 机器人线速度 vx，取世界坐标系 X 方向
        self.robot_x_velocity = base_vel[:, 0]   # 保持tensor

        # 用轮子差分推算位置 (可选，不建议用于真实正解)
        self.robot_x_position = (self.position_lw - self.position_rw) * 0.1 / 2.0

    # ================== [新增] 三大通用接口 ==================
    def init_params(self, registry):
        """
        初始化：只加载注册表里的'模糊参数'，不管 PID。
        """
        # print(f"\n[LQR] 初始化模糊参数... (Total: {len(registry)})")
        
        for name, cfg in registry.items():
            # 1. 过滤掉 VMC 参数 (由 VMC 控制器处理)
            if name.startswith("vmc_"):
                continue
            
            # 2. 加载默认值到 self.params 字典
            # 这些参数 (gamma, eth, ld...) 会在 ST_SIT2_FLC_FM 函数中被直接读取
            default_val = float(cfg["default"])
            self.params[name] = torch.full((self.num_envs,), default_val, device=self.device)
            self.defaults[name] = default_val # 存个备份用于 reset

        # 3. 立即更新观测张量
        self._update_debug_params()    

    def update_params(self, new_params_dict):
        """
        更新：RL 每一步输出动作后调用。
        只更新字典数值，不需要同步到底层对象 (因为底层对象已经写死或者是直接读字典的)。
        """
        for name, value in new_params_dict.items():
            # 只更新属于 LQR 的模糊参数
            if name in self.params:
                self.params[name] = value
        
        self._update_debug_params()

    def reset_params(self, envs_idx):
        """
        重置：倒地后，将模糊参数恢复为初始默认值。
        """
        if len(envs_idx) == 0:
            return

        for name, default_val in self.defaults.items():
            if name in self.params:
                # 仅恢复指定环境的参数
                self.params[name][envs_idx] = default_val
                
        self._update_debug_params()

    def _update_debug_params(self):
        # 如果还没初始化完（防止报错）
        if 'gamma_d_g' not in self.params: 
            return

        # 堆叠 5 个模糊参数
        self.current_fuzzy_params = torch.stack([
            self.params['gamma_d_g'], 
            self.params['eth'], 
            # self.params['ld_max'],
            # self.params['ld_min'], 
            self.params['a'], 
            self.params['k_out']
        ], dim=1)

    def reset_params(self, envs_idx):
        """
        [补全] 重置：倒地后，将模糊参数恢复为初始默认值。
        """
        if len(envs_idx) == 0:
            return

        # 遍历所有记录在案的默认值
        for name, default_val in self.defaults.items():
            if name in self.params:
                # 仅恢复指定环境 (envs_idx) 的参数值
                self.params[name][envs_idx] = default_val
        
        # 别忘了更新观测张量，否则 Actor 看到的还是旧参数
        self._update_debug_params()

    def wrap_to_pi(self, angles):
    # 这是标准的 wrap to pi 逻辑
        return (angles + torch.pi) % (2 * torch.pi) - torch.pi



    def get_command_pitch(self):
        # pitch_d 已由外部 set_commands 设置
        L, M, self.pitch_com, self.roll_com, Jz, Jy = self.com(
            self.right_hip_position,
            self.right_knee_position,
            self.left_hip_position,
            self.left_knee_position,
            self.roll
        )



        # # [修改点] 实时将 params 里的参数灌入 PID
        # self.command_pitch.kp = self.params['pitch_kp']
        # self.command_pitch.kd = self.params['pitch_kd']
        pitch_ref_from_vel = self.get_pitch_ref_from_velocity()

        # pitch_d_1 = self.pitch_d * ( torch.pi / 180 ) #要转换成弧度
        # pitch_d   = self.pitch_com - pitch_ref_from_vel

        # 安全提取 RL 传来的俯仰角偏置。如果没有（如初始化时），默认为 0
        # rl_pitch_offset = self.params.get('rl_pitch_offset', torch.zeros_like(pitch_ref_from_vel))
        
        # 将底层速度环算出的目标，加上 RL 高层给的偏移量
        pitch_d = pitch_ref_from_vel  

        self.theta_pitch = pitch_d - self.pitch
        theta_pitch_pass = self.ST_SIT2_FLC_FM(self.theta_pitch)
        command = self.command_pitch.compute(error=theta_pitch_pass, dt=0.002)
        return command  # shape: (num_envs,)

    def get_pitch_ref_from_velocity(self):
        theta_velocity = self.velocity_d - self.robot_x_velocity
        is_want_to_stop = (torch.abs(self.velocity_d) < 0.002)
    
        # # [修改点] 实时更新 PID
        # self.command_velocity.kp = self.params['vel_kp']
        # self.command_velocity.ki = self.params['vel_ki']
        # self.command_velocity.kd = self.params['vel_kd']
        self.pitch_offset = self.command_velocity.compute(error=theta_velocity, dt=0.002, stopping_mask=is_want_to_stop)
        
        self.pitch_offset = torch.clamp(self.pitch_offset,
        -20.0 * math.pi / 180,
         20.0 * math.pi / 180
        )

        return self.pitch_offset

    def get_command_yaw(self):
        # 1. 角度归一化
        raw_error = self.yaw_d - self.yaw
        theta_yaw = (raw_error + math.pi) % (2 * math.pi) - math.pi
        
        # 2. 计算 PID (output 现在是一个 GPU Tensor)
        output = self.command_yaw.compute(error=theta_yaw, dt=0.002)
        
        # 3. 【核心修复】：使用 torch.clamp 替代 np.clip
        MAX_YAW_TORQUE = 3.0
        # 修改前: output = np.clip(output, -MAX_YAW_TORQUE, MAX_YAW_TORQUE)
        output = torch.clamp(output, -MAX_YAW_TORQUE, MAX_YAW_TORQUE)
        
        return output

    def com(self, xita_r1, xita_r2, xita_l1, xita_l2, roll):
        # 保持原先角度偏置逻辑
        xita_r1 = xita_r1 - 20 * torch.pi / 180
        xita_r2 = xita_r2 - 95 * torch.pi / 180
        xita_l1 = -xita_l1 - 20 * torch.pi / 180
        xita_l2 = -xita_l2 - 95 * torch.pi / 180

        # 常量部分保持不变（转 torch 并放到与输入相同的 device）
        device = xita_r1.device
        lr1 = 150e-3
        lr2 = 250e-3
        ll1 = lr1
        ll2 = lr2
        dlr = 355e-3

        dr1 = torch.tensor([0.020031, 0.0032153, 0.0036019], device=device)
        dr2 = torch.tensor([0.072881, 0.034135, 0.019602], device=device)
        dl1 = torch.tensor([0.019888, 0.0032269, -0.0036874], device=device)
        dl2 = torch.tensor([0.073104, 0.034096, -0.019634], device=device)
        db  = torch.tensor([0.01131, -0.0013618, -0.00028518], device=device)

        mr1 = 0.51378
        mr2 = 0.27188
        ml1 = 0.51183
        ml2 = 0.27057
        mb  = 2.9637

        # 批量 trigonometric 运算
        cr1 = torch.cos(xita_r1)
        sr1 = torch.sin(xita_r1)
        cl1 = torch.cos(xita_l1)
        sl1 = torch.sin(xita_l1)

        cr1r2 = torch.cos(xita_r1 + xita_r2)
        sr1r2 = torch.sin(xita_r1 + xita_r2)
        cl1l2 = torch.cos(xita_l1 + xita_l2)
        sl1l2 = torch.sin(xita_l1 + xita_l2)

        def _expand_z(val, ref): 
            return torch.full_like(ref, val)

        Xr1 = torch.stack([
            cr1 * dr1[0] - sr1 * dr1[1],
            sr1 * dr1[0] + cr1 * dr1[1],
            _expand_z(dr1[2] + dlr / 2, xita_r1)
        ], dim=-1)

        Xr2 = torch.stack([
            cr1r2 * dr2[0] - sr1r2 * dr2[1] + lr1 * cr1,
            sr1r2 * dr2[0] + cr1r2 * dr2[1] + lr1 * sr1,
            _expand_z(dr2[2] + dlr / 2, xita_r1)
        ], dim=-1)

        Xl1 = torch.stack([
            cl1 * dl1[0] - sl1 * dl1[1],
            sl1 * dl1[0] + cl1 * dl1[1],
            _expand_z(dl1[2] - dlr / 2, xita_r1)
        ], dim=-1)

        Xl2 = torch.stack([
            cl1l2 * dl2[0] - sl1l2 * dl2[1] + ll1 * cl1,
            sl1l2 * dl2[0] + cl1l2 * dl2[1] + ll1 * sl1,
            _expand_z(dl2[2] - dlr / 2, xita_r1)
        ], dim=-1)

        Xb = torch.stack([
            _expand_z(db[0], xita_r1),
            _expand_z(db[1], xita_r1),
            _expand_z(db[2], xita_r1)
        ], dim=-1)

        # 质心计算
        Xbody = (mr1 * Xr1 + mr2 * Xr2 + ml1 * Xl1 + ml2 * Xl2 + mb * Xb) / (mr1 + mr2 + ml1 + ml2 + mb)

        T_80_3 = torch.stack([
            (lr2 * torch.cos(xita_r1 + xita_r2) + lr2 * torch.cos(xita_l1 + xita_l2)
            + lr1 * torch.cos(xita_l1) + lr1 * torch.cos(xita_r1)) / 2,
            (lr2 * torch.sin(xita_l1 + xita_l2) + lr2 * torch.sin(xita_r1 + xita_r2)
            + lr1 * torch.sin(xita_l1) + lr1 * torch.sin(xita_r1)) / 2,
            torch.zeros_like(xita_r1)
        ], dim=-1)

        X8_re = Xbody - T_80_3

        # 机体 roll 变换
        c = torch.cos(-roll)
        s = torch.sin(-roll)
        Rx_roll = torch.stack([
            torch.stack([torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)], dim=-1),
            torch.stack([torch.zeros_like(c), c, -s], dim=-1),
            torch.stack([torch.zeros_like(c), s,  c], dim=-1)
        ], dim=-2)

        X8_re = torch.matmul(Rx_roll, X8_re.unsqueeze(-1)).squeeze(-1)

        pitch_com = torch.atan(X8_re[:, 0] / X8_re[:, 1])
        roll_com  = torch.atan(X8_re[:, 2] / X8_re[:, 1])

        L = torch.sqrt(X8_re[:, 0]**2 + X8_re[:, 1]**2)

        # 转动惯量等效质量计算
        def dist(a, b):
            return torch.sqrt(torch.sum((a - b) ** 2, dim=-1))

        rr1 = dist(Xr1, T_80_3)
        rr2 = dist(Xr2, T_80_3)
        rl1 = dist(Xl1, T_80_3)
        rl2 = dist(Xl2, T_80_3)
        rb  = dist(Xb,  T_80_3)

        m = (mr2*rr2**2 + ml2*rl2**2 + mr1*rr1**2 + ml1*rl1**2 + mb*rb**2) / (L**2)
        J_z = m * L**2

        # roll 惯量
        def dist_r(a, b):
            return torch.sqrt((a[:, 0] - b[:, 0])**2 + (a[:, 2] - b[:, 2])**2)

        rr1_r = dist_r(Xr1, T_80_3)
        rr2_r = dist_r(Xr2, T_80_3)
        rl1_r = dist_r(Xl1, T_80_3)
        rl2_r = dist_r(Xl2, T_80_3)
        rb_r  = dist_r(Xb,  T_80_3)

        J_y = mr2*rr2_r**2 + ml2*rl2_r**2 + mr1*rr1_r**2 + ml1*rl1_r**2 + mb*rb_r**2

        return L, m, pitch_com, roll_com, J_z, J_y


    def ST_SIT2_FLC_FM(self, error_pass: torch.Tensor):
        # =========================================================
        # 1. 锁死安全论域 35 度，绝不轻易溢出
        # =========================================================
        DOMAIN_DEG = 35.0
        error = error_pass / math.pi * 180.0 / DOMAIN_DEG
        MAX_RAD = DOMAIN_DEG / 180.0 * math.pi

        # =========================================================
        # 2. 读取 RL 动态参数 (不再读取 ld_max 和 ld_min)
        # =========================================================
        gamma_d_g = self.params.get('gamma_d_g', torch.full_like(error, 45*math.pi/180))
        eth       = self.params.get('eth', torch.full_like(error, 0.06))
        a         = self.params.get('a', torch.full_like(error, 0.5))
        k_out     = self.params.get('k_out', torch.ones_like(error)) # 默认输出增益为1.0
        
        ld_max    = 0.9  # 坚决写死，防止数学奇点
        ld_min    = 0.3

        # =========================================================
        # 3. 动态数学保护屏障
        # =========================================================
        cos_gamma = torch.cos(gamma_d_g)
        max_safe_eth = cos_gamma - 0.001
        eth = torch.minimum(eth, max_safe_eth)

        ld_g = (torch.cos(gamma_d_g) - eth) / torch.cos(gamma_d_g)

        B0_g = torch.zeros_like(ld_g)
        B1_g = ld_g * torch.sin(gamma_d_g)
        B2_g = torch.ones_like(ld_g)

        C0_g = torch.zeros_like(ld_g)
        C1_g = ld_g * torch.cos(gamma_d_g)
        C2_g = torch.ones_like(ld_g)

        K0_g = (B1_g - B0_g) / (C1_g - C0_g)
        K1_g = (B2_g - B1_g) / (C2_g - C1_g)
        N0_g = (B0_g * C1_g - B1_g * C0_g) / (C1_g - C0_g)
        N1_g = (B1_g * C2_g - B2_g * C1_g) / (C2_g - C1_g)

        # ---------- gamma_g 分段计算 ----------
        cond1 = (error <= -C2_g)
        cond2 = (error > -C2_g) & (error < -C1_g)
        cond3 = (error >= -C1_g) & (error < C1_g)
        cond4 = (error >= C1_g) & (error < C2_g)
        cond5 = (error >= C2_g)

        gamma_g = torch.zeros_like(error)
        gamma_g = torch.where(cond1, B2_g * 70 + 10, gamma_g)
        gamma_g = torch.where(cond2, (K1_g * torch.abs(error) + N1_g) * 70 + 10, gamma_g)
        gamma_g = torch.where(cond3, (K0_g * torch.abs(error)) * 70 + 10, gamma_g)
        gamma_g = torch.where(cond4, (K1_g * error + N1_g) * 70 + 10, gamma_g)
        gamma_g = torch.where(cond5, B2_g * 70 + 10, gamma_g)

        # ---------- ld_e 与 比例系数 ----------
        gamma_g_1 = gamma_g * math.pi / 180
        ld_e = torch.abs(error) / torch.cos(gamma_g_1)
        ld_e = torch.clamp(ld_e, ld_min, ld_max)
        ld = ld_e

        B0 = torch.zeros_like(ld)
        B1 = ld * torch.sin(gamma_g_1)
        B2 = torch.ones_like(ld)
        C0 = torch.zeros_like(ld)
        C1 = ld * torch.cos(gamma_g_1)
        C2 = torch.ones_like(ld)

        m0 = a; m1 = 1 - a; m2 = a
        eps = 1e-8

        K0 = 0.5 * ((B1 - B0 * m0) / (C1 * m0 - C0 + torch.abs(error) * (-m0 + 1) + eps) +
                    (B0 - B1 * m1) / (C0 * m1 - C1 + torch.abs(error) * (-m1 + 1) + eps))
        K1 = 0.5 * ((B2 - B1 * m1) / (C2 * m1 - C1 + torch.abs(error) * (-m1 + 1) + eps) +
                    (B1 - B2 * m2) / (C1 * m2 - C2 + torch.abs(error) * (-m2 + 1) + eps))
        N0 = 0.5 * ((B1 * C0 - B0 * C1 * m0) / (-C1 * m0 + C0 + torch.abs(error) * (m0 - 1) + eps) +
                    (B0 * C1 - B1 * C0 * m1) / (-C0 * m1 + C1 + torch.abs(error) * (m1 - 1) + eps))
        N1 = 0.5 * ((B2 * C1 - B1 * C2 * m1) / (-C2 * m1 + C1 + torch.abs(error) * (m1 - 1) + eps) +
                    (B1 * C2 - B2 * C1 * m2) / (-C1 * m2 + C2 + torch.abs(error) * (m2 - 1) + eps))

        # =========================================================
        # 4. φ 分段计算 (彻底修复量纲：统一乘以 MAX_RAD)
        # =========================================================
        phi = torch.zeros_like(error)
        phi = torch.where(cond1, (-B2 * MAX_RAD) + (error + C2) * (K1 * MAX_RAD), phi)
        phi = torch.where(cond2, (K1 * error - N1) * MAX_RAD, phi)
        phi = torch.where(cond3, (K0 * error) * MAX_RAD, phi)
        phi = torch.where(cond4, (K1 * error + N1) * MAX_RAD, phi)
        phi = torch.where(cond5, (B2 * MAX_RAD) + (error - C2) * (K1 * MAX_RAD), phi)

        # 乘以 RL 算出的终极爆发力增益
        return phi * k_out
    
    def balance(self,):
        a = self.get_command_pitch()
        # b = self.get_command_velocity()
        c = self.get_command_yaw()
        left_command = -0.5*a-c
        right_command = 0.5*a-c
        left_command  = left_command.to(dtype=torch.float32)
        right_command = right_command.to(dtype=torch.float32)
        wheel_forces = torch.stack([left_command, right_command], dim=1)

        return wheel_forces


class BatchedVMC:
    def __init__(self, num_envs: int, device: torch.device = None):
        self.num_envs = int(num_envs)
        self.device = torch.device('cpu') if device is None else device
        # self.robot = robot
        self.dt = 0.002
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.joint_name_to_idx = None
        # [核心修复] 在这里直接初始化！不要等 update_debug_params
        # 3 是因为 VMC 有 2/3 个参数 (z_l, z_r, k（可选）)
        self.current_vmc_params = torch.zeros((self.num_envs, 2), device=self.device)
        # [重构] 同样加入通用参数字典
        self.params = {}
        self.defaults = {}
        # [新增] 存储基准高度指令，默认值设为你的标准站立高度
        self.command_height = torch.full((self.num_envs,), 0.2779, device=self.device)
        # ========= 向量化创建工具 =========
        def z():
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # ================== 关节位置（全并行张量化） ==================
        self.position_lh = z()  # 左髋
        self.position_lk = z()  # 左膝
        self.position_rh = z()  # 右髋
        self.position_rk = z()  # 右膝
        self.position_lw = z()  # 左轮
        self.position_rw = z()  # 右轮

        # ================== 关节速度 ==================
        self.velocity_lh = z()
        self.velocity_lk = z()
        self.velocity_rh = z()
        self.velocity_rk = z()
        self.velocity_lw = z()
        self.velocity_rw = z()

        # ================== IMU 姿态 ==================
        self.roll     = z()   # shape = (num_envs,)
        self.roll_dot = z()   # shape = (num_envs,)

        # 四元数 shape = (num_envs, 4)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

        # ================== 关节索引映射（与 motor_dofs 对应） ==================
        # self.joint_names = [
        #     "L1_joint", "L2_joint", "R1_joint", "R2_joint", "L3_joint", "R3_joint"
        # ]
        
        # self.joint_name_to_idx = {
        #     "left_wheel":  11, # motor_dofs[4]
        #     "right_wheel": 10, # motor_dofs[5]
        #     "left_hip":    7,  # motor_dofs[0]
        #     "left_knee":   9,  # motor_dofs[1]
        #     "right_hip":   6,  # motor_dofs[2]
        #     "right_knee":  8   # motor_dofs[3]
        # }
    def set_joint_mapping(self, joint_name_to_idx):
        """设置关节索引映射"""
        self.joint_name_to_idx = joint_name_to_idx        

    def set_state(self, dof_pos, dof_vel, base_vel, base_euler, ang):
        """从环境接收状态"""
        dof_pos = dof_pos.to(self.device)
        dof_vel = dof_vel.to(self.device)
        base_vel = base_vel.to(self.device)
        base_euler = base_euler.to(self.device)

        # quat = quat.to(self.device)
        ang = ang.to(self.device)
        
        # 提取关节状态
        self.position_lw = dof_pos[:, self.joint_name_to_idx["left_wheel"]]
        self.position_rw = dof_pos[:, self.joint_name_to_idx["right_wheel"]]
        self.position_lh = dof_pos[:, self.joint_name_to_idx["left_hip"]]
        self.position_rh = dof_pos[:, self.joint_name_to_idx["right_hip"]]
        self.position_lk = dof_pos[:, self.joint_name_to_idx["left_knee"]]
        self.position_rk = dof_pos[:, self.joint_name_to_idx["right_knee"]]
        
        self.velocity_lw = dof_vel[:, self.joint_name_to_idx["left_wheel"]]
        self.velocity_rw = dof_vel[:, self.joint_name_to_idx["right_wheel"]]
        self.velocity_lh = dof_vel[:, self.joint_name_to_idx["left_hip"]]
        self.velocity_rh = dof_vel[:, self.joint_name_to_idx["right_hip"]]
        self.velocity_lk = dof_vel[:, self.joint_name_to_idx["left_knee"]]
        self.velocity_rk = dof_vel[:, self.joint_name_to_idx["right_knee"]]
        
        # 更新 IMU

        # self.base_quat = quat
        
        # self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # aligned_quat = transform_quat_by_quat(
        #     torch.ones_like(self.base_quat) * self.inv_base_init_quat,
        #     self.base_quat
        # )
        # rpy = quat_to_xyz(aligned_quat)
        # self.roll = rpy[:, 0]

        self.roll_dot = ang[:, 0] * torch.pi/180

        euler = base_euler
        self.roll = euler[:, 0] * torch.pi/180
        self.pitch = euler[:, 1]
        self.yaw = euler[:, 2]
    
    def set_commands(self, height_target=None):
        """接收来自环境的宏观指令 (User Command)"""
        if height_target is not None:
            self.command_height = height_target
# # [新增] 同样的通用接口
#     def init_params(self, registry):
#         # 筛选出以 "vmc_" 开头的参数，或者简单点全部初始化
#         for name, cfg in registry.items():
#             # [新增] VMC 控制器只关心 vmc_ 开头的参数
#             if not name.startswith("vmc_"):
#                 continue
#             default_val = float(cfg["default"])
#             self.params[name] = torch.full((self.num_envs,), default_val, device=self.device)
#             self.defaults[name] = default_val
#         # [新增] 初始化完成后，立刻生成第一次 current_vmc_params，防止后续报错
#         self._update_debug_params()    

#     def update_params(self, new_params_dict):
#         for name, value in new_params_dict.items():
#             if name in self.params:
#                 self.params[name] = value
#         self._update_debug_params()

#     def _update_debug_params(self):
#         """
#         VMC 控制器：只负责输出 3 个物理参数
#         """
#         # [修正] 只堆叠 VMC 相关的 3 个参数
#         # 并且变量名改为 current_vmc_params
#         self.current_vmc_params = torch.stack([
#             # self.params['vmc_z_l'],
#             # self.params['vmc_z_r'],
#             # self.params['vmc_k'],
#             # self.params['vmc_d'],
#             # self.params['vmc_b_l'],
#             # self.params['vmc_b_r']
#             self.params["vmc_roll_k"],
#             self.params["vmc_roll_d"]
#         ], dim=1)   

    
                
#     def reset_params(self, envs_idx):
#         if len(envs_idx) == 0: return
#         for name, val in self.defaults.items():
#             self.params[name][envs_idx] = val
#         self._update_debug_params()

    def vmc(self):
        # ================== 原始标量参数，保持不变 ==================
        k = 400  # 120
        d = 40
        theta_10 = 70 * math.pi / 180
        theta_20 = 95 * math.pi / 180
        l1 = 0.15
        l2 = 0.25
        x_d_hip = 0.0353   # 足端相对于 Hip 的期望 X 偏移
        x_dot_d = 0.0
        z_dot_d = 0.0
        z_d = 0.2779

        # # (A) 刚度 Stiffness K
        # stiffness_k = self.params.get('vmc_k', torch.full((self.num_envs,), 400.0, device=self.device))
        # damping_d = self.params.get('vmc_d', torch.full((self.num_envs,), 40.0, device=self.device))
        # bias_l = self.params.get('vmc_b_l', torch.zeros((self.num_envs,), device=self.device))
        # bias_r = self.params.get('vmc_b_r', torch.zeros((self.num_envs,), device=self.device))


        # # 默认值必须是 0.0！表示RL不干预时，不产生额外伸缩
        # delta_l = self.params.get('vmc_z_l', torch.zeros((self.num_envs,), device=self.device))
        # delta_r = self.params.get('vmc_z_r', torch.zeros((self.num_envs,), device=self.device))

        # ================== 张量化 roll 力补偿 ==================
            # roll_k = self.params.get('vmc_roll_k', torch.full((self.num_envs,), 2500.0, device=self.device))
            # roll_d = self.params.get('vmc_roll_d', torch.full((self.num_envs,), 100.0, device=self.device))

        z_d_l =( 2500 * self.roll + 100 * self.roll_dot )  # (num_envs,) 2500 , 100

        # ================== 连杆位置 ==================
        # 注意：sin/cos 用 torch 版本，不改变表达式结构
        x_l = l1 * torch.sin(theta_10 - self.position_lh) \
            - l2 * torch.sin((theta_20 + self.position_lk) - (theta_10 - self.position_lh))

        self.real_z_l = l1 * torch.cos(theta_10 - self.position_lh) \
            + l2 * torch.cos((theta_20 + self.position_lk) - (theta_10 - self.position_lh))

        x_r = l1 * torch.sin(theta_10 + self.position_rh) \
            - l2 * torch.sin((theta_20 - self.position_rk) - (theta_10 + self.position_rh))

        self.real_z_r = l1 * torch.cos(theta_10 + self.position_rh) \
            + l2 * torch.cos((theta_20 - self.position_rk) - (theta_10 + self.position_rh))

        # ================== 雅可比矩阵（矢量化） ==================
        # shape: (num_envs, 2, 2)
        jcobi_l = torch.stack([
            torch.stack([
                -l1 * torch.cos(theta_10 - self.position_lh)
                - l2 * torch.cos((theta_20 + self.position_lk) - (theta_10 - self.position_lh)),

                -l2 * torch.cos((theta_20 + self.position_lk) - (theta_10 - self.position_lh))
            ], dim=1),
            torch.stack([
                l1 * torch.sin(theta_10 - self.position_lh)
                - l2 * torch.sin((theta_20 + self.position_lk) - (theta_10 - self.position_lh)),

                -l2 * torch.sin((theta_20 + self.position_lk) - (theta_10 - self.position_lh))
            ], dim=1)
        ], dim=1)

        jcobi_r = torch.stack([
            torch.stack([
                l1 * torch.cos(theta_10 + self.position_rh)
                + l2 * torch.cos((theta_20 - self.position_rk) - (theta_10 + self.position_rh)),

                l2 * torch.cos((theta_20 - self.position_rk) - (theta_10 + self.position_rh))
            ], dim=1),
            torch.stack([
                -l1 * torch.sin(theta_10 + self.position_rh)
                + l2 * torch.sin((theta_20 - self.position_rk) - (theta_10 + self.position_rh)),

                l2 * torch.sin((theta_20 - self.position_rk) - (theta_10 + self.position_rh))
            ], dim=1)
        ], dim=1)

        # ================== 关节角速度 ==================
        theta_dot_l = torch.stack([self.velocity_lh, self.velocity_lk], dim=1).unsqueeze(-1)
        theta_dot_r = torch.stack([self.velocity_rh, self.velocity_rk], dim=1).unsqueeze(-1)

        # ================== 足端速度 ==================
        # p_dot shape -> (num_envs, 2)
        p_dot_l = torch.bmm(jcobi_l, theta_dot_l).squeeze(-1)
        p_dot_r = torch.bmm(jcobi_r, theta_dot_r).squeeze(-1)

        # ================== VMC 力 ==================
        # 左腿力 F_l (使用 target_height_left)
        # 注意: 垂直方向加上了 -z_d_l  

        #！！！！！！！！！这里加上了k*bias，意味补偿力，这样才是真正的阻抗控制，否则就是位置控制！！！！！！！！！！！
        F_l = torch.stack([
            k * (x_d_hip  - x_l)  + d * (x_dot_d - p_dot_l[:, 0]),
            k * (z_d - self.real_z_l ) + d * (z_dot_d - p_dot_l[:, 1]) - z_d_l 
        ], dim=1)

        # 右腿力 F_r (使用 target_height_right)
        # 注意: 垂直方向加上了 +z_d_l
        F_r = torch.stack([
            k * (x_d_hip  - x_r)  + d * (x_dot_d - p_dot_r[:, 0]),
           k * (z_d - self.real_z_r ) + d * (z_dot_d - p_dot_r[:, 1]) + z_d_l 
        ], dim=1)

        # ================== 关节力矩 ==================
        # τ = J^T F
        tao_l = torch.bmm(jcobi_l.transpose(1, 2), F_l.unsqueeze(-1)).squeeze(-1)
        tao_r = torch.bmm(jcobi_r.transpose(1, 2), F_r.unsqueeze(-1)).squeeze(-1)

        force = torch.stack([
            tao_l[:, 0],
            tao_l[:, 1],
            tao_r[:, 0],
            tao_r[:, 1],
        ], dim=1)  # shape (num_envs, 4)

        return force

def main():
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import argparse
    import os
    import sys
    import time
    import math
    from Pitch_env import WheelLeggedEnv
    import torch

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    parser = argparse.ArgumentParser(description="测试模糊控制器")
    parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量")
    parser.add_argument("--use_keyboard", action="store_true", help="使用键盘控制")
    parser.add_argument("--test_vel", type=float, default=1.0, help="自动化测试目标速度 (m/s)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"    模糊控制器极简测试 | 目标速度: {args.test_vel} m/s")
    print("="*60 + "\n")
    
    # ===============================================================
    # 【核心修改 1】：抛弃 GPU！单环境测试必须用 CPU 后端才能消除调度延迟
    # ===============================================================
    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning") 
    
    from Pitch_train import get_cfgs
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
    
    terrain_cfg["terrain"] = True
    terrain_cfg["eval"] = "precise_track" 
    
    print("🔧 初始化环境...")
    env = WheelLeggedEnv(
        num_envs=args.num_envs,
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
        # ===============================================================
        # 【核心修改 2】：告诉环境内所有的 PyTorch 张量，全部在 CPU 上计算
        # ===============================================================
        device="cpu" 
    )
    
    # 初始化 TensorBoard 和其余逻辑保持不变
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(current_dir, "logs", "controller_eval", f"Vel_{args.test_vel}_{timestamp}")
    writer = SummaryWriter(log_dir)
    
    pad = None
    if args.use_keyboard:
        try:
            sys.path.append("/home/zhjkoi/wheel_leg_paper/wheel_legged_genesis/locomotion")
            from utils import gamepad
            commands_scale = [0.1, 0.5, 1.57, 0.28]
            pad = gamepad.control_gamepad(command_cfg, commands_scale)
            print("🎮 键盘控制已启用")
        except Exception as e:
            pass

    obs, _ = env.reset()
    
    rl_dt = env.dt * env.decimation  # 0.02s
    current_sim_time = 0.0   
    current_step = 0         
    max_test_time = 100     
    
    finish_end = getattr(env, 'finish_line_x_end', 12.8)
    finish_start = getattr(env, 'finish_line_x_start', -0.5)

    print("\n" + "="*60)
    print("🚀 CPU 高速测试循环启动 (按 Ctrl+C 退出)")
    print("="*60 + "\n")
    real_start_time = time.time()
    try:
        with torch.no_grad():
            while current_sim_time < max_test_time:
                loop_start = time.perf_counter()

                # ===============================================================
                # 【核心修改 3】：这里输入给 step 的 actions 也必须在 cpu 上
                # ===============================================================
                actions = torch.zeros((args.num_envs, env.num_actions), device="cpu")
                
                obs, _, rews, dones, infos = env.step(actions)
                
                # 指令分发
                if pad is not None:
                    commands, reset_flag = pad.get_commands()
                    current_command = commands
                    env.set_commands([0], current_command)
                    if reset_flag:
                        env.reset()
                        current_step = 0
                        current_sim_time = 0.0
                        continue
                else:
                    current_command = [args.test_vel, 0.0, 1.57, 0.28]
                    env.set_commands(0, current_command)

                # 更新时间
                current_step += 1
                current_sim_time = current_step * rl_dt

                # 指标提取
                env_id = 0
                vel_x_real = env.base_lin_vel[env_id, 0].item()
                vel_x_cmd = current_command[0]
                
                vel_error = vel_x_cmd - vel_x_real
                pitch_deg = env.base_euler[env_id, 1].item() * (180 / math.pi)
                pitch_dot = env.base_ang_vel[env_id, 1].item()
                roll_deg = env.base_euler[env_id, 0].item() * (180 / math.pi)
                
                writer.add_scalar('Metrics/1_Vel_Error', vel_error, current_step)
                writer.add_scalar('Metrics/2_Pitch_Error', pitch_deg, current_step)
                writer.add_scalar('Metrics/3_Pitch_Dot', pitch_dot, current_step)
                writer.add_scalar('Metrics/4_Roll', roll_deg, current_step)
                
                # 冲线判断
                current_x = env.base_pos[env_id, 0].item()
                if not args.use_keyboard:
                    if vel_x_cmd >= 0 and current_x > (finish_end + 0.5):
                        print(f"\n🏆 前进冲线成功！停止测试，当前位置: X = {current_x:.2f}m")
                        break
                    elif vel_x_cmd < 0 and current_x < (finish_start - 0.5):
                        print(f"\n🏆 倒车冲线成功！停止测试，当前位置: X = {current_x:.2f}m")
                        break
                real_elapsed = time.time() - real_start_time
                # if current_step % 10 == 0:
                #         print(f"仿真时间: {current_sim_time:.4f}s | 现实时间: {real_elapsed:.4f}s")
                if current_step % int(1.0 / rl_dt) == 0:
                    print(f"Time: {current_sim_time:.4f}s | Cmd: {vel_x_cmd:.4f} | Vel_x: {vel_x_real:.4f} | Pitch: {pitch_deg:.4f}°")
                    print(f"pitch: {env.lqr_controller.pitch}, yaw: {env.lqr_controller.yaw}, roll: {env.lqr_controller.roll}")
                    print(f"pitch_dot: {env.lqr_controller.pitch_dot}, velocity: {env.lqr_controller.robot_x_velocity}")
                    print(f"vel_error: {env.lqr_controller.velocity_d - env.lqr_controller.robot_x_velocity}")
                    print(f"pitch_offset: {env.lqr_controller.pitch_offset}")
                    print(f"command: {env.commands}")
                    print(f"acc_x: {env.base_lin_acc}")
                    print(f"projected_gravity: {env.projected_gravity}")
                    # print(f"step_time: {step_time:.4f} sec")
                    print("====================================================================")
                
                # 挂钟同步 (在 CPU 模式下，它的计算速度远超 0.02s，所以这步必须有)
                elapsed = time.perf_counter() - loop_start
                if elapsed < rl_dt:
                    time.sleep(rl_dt - elapsed)
                
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⏹️  测试被手动中断")
    finally:
        writer.close() 
        print("="*60)
        print(f"✅ 测试结束，数据已保存。")

if __name__ == "__main__":
    main()