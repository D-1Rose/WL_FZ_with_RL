
import math
from transforms3d import euler
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat # type: ignore
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import torch
import time
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

#左轮lqr控制
class LQR_Controller:
    def __init__(self,robot):
        # rospy.init_node('LQR_Controller')
        # self.rate = rospy.Rate(500) # 节点执行的频率，单位赫兹
        # 初始化变量来存储左轮和右轮的速度和位置
        self.robot = robot
        self.dt = 0.002
        self.base_quat = [0,0,0,0]

#=========================================================================
        # 模糊参数默认值（沿用原硬编码）
        self.fuzzy_gamma_d_g_default = 75 * np.pi / 180  # 75°（弧度）
        self.fuzzy_eth_default = 0.06
        self.fuzzy_ld_max_default = 0.9
        self.fuzzy_ld_min_default = 0.3
        self.fuzzy_a_default = 0.1

        # 定义最大偏移量delta（根据控制需求设定，确保参数范围合理）
        self.gamma_d_g_delta = 25 * np.pi / 180  # 允许±25°（范围：50°~100°）
        self.eth_delta = 0.04  # 允许±0.04（范围：0.02~0.10）
        self.ld_max_delta = 0.3  # 允许±0.3（范围：0.6~1.2）
        self.ld_min_delta = 0.1  # 允许±0.1（范围：0.2~0.4）
        self.a_delta = 0.05  # 允许±0.05（范围：0.05~0.15）

        # 初始参数值=默认值（未偏移）
        self.fuzzy_gamma_d_g = self.fuzzy_gamma_d_g_default
        self.fuzzy_eth = self.fuzzy_eth_default
        self.fuzzy_ld_max = self.fuzzy_ld_max_default
        self.fuzzy_ld_min = self.fuzzy_ld_min_default
        self.fuzzy_a = self.fuzzy_a_default
#=========================================================================






        
        self.left_wheel_velocity = 0.0 #左轮
        # self.left_wheel_position = 1.0
        self.left_wheel_position = 0.0        
        self.right_wheel_velocity = 0.0 #右轮
        self.right_wheel_position = 0.0
        self.left_hip_position = 0.0
        self.right_hip_position = 0.0
        self.left_knee_position = 0.0
        self.right_knee_position = 0.0
        self.left_hip_effort = 0.0
        self.left_knee_effort = 0.0
        self.left_wheel_effort = 0.0
        self.right_hip_effort = 0.0
        self.right_knee_effort = 0.0
        self.right_wheel_effort = 0.0
        self.roll=0.0
        self.pitch=0.0
        self.pitch_com=0.0
        self.n=0
        self.pitch_dot=0.0
        self.left_knee_velocity = 0  
        self.right_knee_velocity = 0
        self.left_hip_velocity = 0 
        self.right_hip_velocity = 0

        ###
        self.theta_r1 = 0
        self.theta_r2 = 0
        self.theta_r3 = 0
        self.theta_rb = 0
        self.theta_r0_dot = 0
        self.theta_r1_dot = 0
        self.theta_r2_dot = 0
        self.theta_r3_dot = 0
        self.theta_rb_dot = 0
        self.theta_l1 = 0
        self.theta_l2 = 0
        self.theta_l3 = 0
        self.theta_lb = 0
        self.theta_l0_dot = 0
        self.theta_l1_dot = 0
        self.theta_l2_dot = 0
        self.theta_l3_dot = 0
        self.theta_lb_dot = 0
        self.tao_rh = 0
        self.tao_rk = 0
        self.tao_rw = 0
        self.tao_lh = 0
        self.tao_lk = 0
        self.tao_lw = 0
        self.error_pitch = 0
        self.error_pitch_dot = 0
        self.desired_state = 0 # 初始期望状态
        self.prev_error = 0.0
        self.integral = 0.0
        self.robot_x_velocity =0
        self.integral_yaw = 0.0
        self.yaw_dot =0.0
        self.yaw =0.0
        self.robot_x_position =0.0
        self.velocity_d = 0.0
        self.roll_dot = 0.0

        self.command_yaw = PIDController(kp=-0.0, ki=0, kd=-0.02)#kp=5, ki=0, kd=0.03
        self.command_pitch = PIDController(kp=-550, ki=0, kd=-30)#435 140    #kp=25,ki=0.0,kd=0.5kp=109.60223629, ki=0, kd=10.61355606
        self.command_velocity = PIDController(kp=65, ki=0/200, kd=-4)#kp=25,ki=0.0,kd=0.5
        # self.command_velocity = PIDController(kp=-7.55, ki=-7.55/200, kd=0.05)#kp=25,ki=0.0,kd=0.5


        # self.joint_names = [
        #     "left_thigh_joint",    # 0: 左髋
        #     "left_calf_joint",     # 1: 左膝
        #     "right_thigh_joint",   # 2: 右髋
        #     "right_calf_joint",    # 3: 右膝
        #     "left_wheel_joint",    # 4: 左轮
        #     "right_wheel_joint"    # 5: 右轮
        # ]
        self.joint_names = [
            "L1_joint",    # 0: 左髋
            "L2_joint",     # 1: 左膝
            "R1_joint",   # 2: 右髋
            "R2_joint",    # 3: 右膝
            "L3_joint",    # 4: 左轮
            "R3_joint"    # 5: 右轮
        ]
        self.joint_name_to_idx = {
            "left_wheel": 11,    # 对应motor_dofs[4]
            "right_wheel": 10,    # 对应motor_dofs[5]
            "left_hip": 7,       # 对应motor_dofs[0]
            "left_knee": 9,      # 对应motor_dofs[1]
            "right_hip": 6,      # 对应motor_dofs[2]
            "right_knee": 8      # 对应motor_dofs[3]
        }
#===============================================================================       
    def set_fuzzy_params(self, gamma_d_g, eth, ld_max, ld_min, a):
        # 限制在[默认值 - delta, 默认值 + delta]范围内
        self.fuzzy_gamma_d_g = np.clip(
            gamma_d_g,
            self.fuzzy_gamma_d_g_default - self.gamma_d_g_delta,
            self.fuzzy_gamma_d_g_default + self.gamma_d_g_delta
        )
        self.fuzzy_eth = np.clip(
            eth,
            self.fuzzy_eth_default - self.eth_delta,
            self.fuzzy_eth_default + self.eth_delta
        )
        self.fuzzy_ld_max = np.clip(
            ld_max,
            self.fuzzy_ld_max_default - self.ld_max_delta,
            self.fuzzy_ld_max_default + self.ld_max_delta
        )
        self.fuzzy_ld_min = np.clip(
            ld_min,
            self.fuzzy_ld_min_default - self.ld_min_delta,
            self.fuzzy_ld_min_default + self.ld_min_delta
        )
        self.fuzzy_a = np.clip(
            a,
            self.fuzzy_a_default - self.a_delta,
            self.fuzzy_a_default + self.a_delta
        )

        self.currunt_fuzzy_params = [self.fuzzy_gamma_d_g, self.fuzzy_eth, self.fuzzy_ld_max, self.fuzzy_ld_min, self.fuzzy_a]
#====================================================================


    #拿到电机反馈关节位置
    def update_joint_states(self):
        # 从消息中提取仿真中的关节速度信息
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        # print(f"self.jopint_names:{self.joint_names}")  # self.jopint_names:['left_thigh_joint', 'left_calf_joint', 'right_thigh_joint', 'right_calf_joint', 'left_wheel_joint', 'right_wheel_joint']
        # print(f"self.motor_dofs:{self.motor_dofs}")  # self.motor_dofs:[7, 9, 6, 8, 11, 10]
        # print(f"self.name:{[self.robot.get_joint(name) for name in self.joint_names]}")

        left_wheel_index = self.motor_dofs[4] # 左轮关节
        right_wheel_index = self.motor_dofs[5] # 右轮关节
        left_hip_index = self.motor_dofs[0]  # 左轮关节
        right_hip_index = self.motor_dofs[2]  # 右轮关节
        left_knee_index = self.motor_dofs[1]  # 左轮关节
        right_knee_index = self.motor_dofs[3]  # 右轮关节

        # print(f"left_wheel_index:{left_wheel_index}")
        # print(f"right_wheel_index:{right_wheel_index}")
        # print(f"left_hip_index:{left_hip_index}")
        # print(f"right_hip_index:{right_hip_index}")
        # print(f"left_knee_index:{left_knee_index}")
        # print(f"right_knee_index:{right_knee_index}")

        # print(f"pos:{self.robot.get_dofs_position()}")
        # ===============useful!!!=============================
        # print(f"left_wheel_index:{left_wheel_index}")
        # print(f"self.position_lw:{self.position_lw}")
        # print(f"self.robot.get_dofs_position():{self.robot.get_dofs_position()}")
        # print(f"self.robot.get_joint:{self.robot.get_joint}")
        # pos = self.robot.get_dofs_position()
        # for name in self.joint_names:
        #     idx = self.robot.get_joint(name).dof_idx_local
        #     print(f"{name}: idx={idx}, pos={pos[idx].item()}")

        self.position_lw = float((self.robot.get_dofs_position()[:,left_wheel_index]))  # 左轮位置（弧度）
        self.position_rw = float((self.robot.get_dofs_position()[:,right_wheel_index]))  # 右轮位置（弧度）
        self.position_lh = float((self.robot.get_dofs_position()[:,left_hip_index])) 
        self.position_rh = float((self.robot.get_dofs_position()[:,right_hip_index]))
        self.position_lk = float((self.robot.get_dofs_position()[:,left_knee_index]))  
        self.position_rk = float((self.robot.get_dofs_position()[:,right_knee_index]))
        ####
        self.velocity_lw = float((self.robot.get_dofs_velocity()[:,left_wheel_index]))  # 左轮速度（弧度/秒）#这里左轮速度是负的，要进行取负处理！
        self.velocity_rw = float((self.robot.get_dofs_velocity()[:,right_wheel_index]))
        self.velocity_lk = float((self.robot.get_dofs_velocity()[:,left_knee_index])) 
        self.velocity_rk = float((self.robot.get_dofs_velocity()[:,right_knee_index]))
        self.velocity_lh = float((self.robot.get_dofs_velocity()[:,left_hip_index]))
        self.velocity_rh = float((self.robot.get_dofs_velocity()[:,right_hip_index]) ) 

        # 力矩（力）
        self.left_wheel_effort = float(self.robot.get_dofs_force()[:,left_wheel_index])
        self.right_wheel_effort = float(self.robot.get_dofs_force()[:,right_wheel_index])
        self.left_hip_effort = float(self.robot.get_dofs_force()[:,left_hip_index])
        self.right_hip_effort = float(self.robot.get_dofs_force()[:,right_hip_index])
        self.left_knee_effort = float(self.robot.get_dofs_force()[:,left_knee_index])
        self.right_knee_effort = float(self.robot.get_dofs_force()[:,right_knee_index])  
        ###动力学模型右腿位置换算
        self.theta_r1 = self.right_wheel_position
        self.theta_r2 = -1*(95/180 * math.pi - self.right_knee_position)
        self.theta_r3 = self.right_hip_position - 20/180 * math.pi 
        self.theta_rb = -1*self.theta_r2 - self.theta_r3 - math.pi/2 + self.pitch
        ##动力学模型右腿速度换算
        self.theta_r1_dot = self.right_wheel_velocity
        self.theta_r2_dot = self.right_knee_velocity
        self.theta_r3_dot = self.right_hip_velocity
        self.theta_rb_dot = -self.theta_r2_dot - self.theta_r3_dot + self.pitch_dot
        ##动力学模型左腿位置换算
        self.theta_l1 = -self.left_wheel_position#####注意
        self.theta_l2 = -1*(95/180 * math.pi + self.left_knee_position)
        self.theta_l3 = -self.left_hip_position - 20/180 * math.pi
        self.theta_lb = -1*self.theta_l2 - self.theta_l3 - math.pi/2 + self.pitch
        ##动力学模型左腿速度换算
        self.theta_l1_dot = -self.left_wheel_velocity
        self.theta_l2_dot = -self.left_knee_velocity
        self.theta_l3_dot = -self.left_hip_velocity
        self.theta_lb_dot = -self.theta_l2_dot - self.theta_l3_dot + self.pitch_dot
        ######
        self.robot_x_position = (self.left_wheel_position - self.right_wheel_position) * 0.1 / 2.0
        # self.robot_x_velocity = (self.left_wheel_velocity + self.right_wheel_velocity) * 0.1/ 2.0
        self.robot_x_velocity = float(self.robot.get_vel()[:,0])

        # print(f"self.robot_position: {self.robot.get_dofs_position()}")
    
        
        # print(f"robot_x_velocity: {self.robot_x_velocity}")
        print(f"pitch: {self.pitch}")
        print(f"yaw: {self.yaw}")

        print(f"robot_velocity: {self.robot.get_vel()}")
        # print(f"shape: {self.robot_x_velocity.shape}")


    def update_imu_data(self):

        w_x = self.robot.get_ang()[:,0]
        w_y = self.robot.get_ang()[:,1]
        w_z = self.robot.get_ang()[:,2]

        # a_x = self.robot.get_acc()[0]
        # a_y = self.robot.get_acc()[1]
        # a_z = self.robot.get_acc()[2]

        quat = self.robot.get_quat()
        if isinstance(quat, torch.Tensor):
            quat = quat.detach().cpu().numpy()
        elif isinstance(quat, (list, tuple)):

            quat = np.array(quat, dtype=np.float32)
 
        w, x, y, z = quat[:,1], quat[:,2], quat[:,3], quat[:,0]

        # 将四元素转化为欧拉角并对实际情况进行数据处理        # rpy_angle 中存放的就是欧拉角，分别是绕 x、y、z 轴的角度
        rpy_angle = euler.quat2euler([w, x, y, z])
        self.roll=rpy_angle[0]
        self.pitch = rpy_angle[1]
        self.yaw = rpy_angle[2]
        # 获取base_link角速度信息
        self.yaw_dot = w_z
        self.pitch_dot = w_y
        self.roll_dot = w_x


    def get_command_pitch(self):
        L, M, self.pitch_com, roll_com,Jz,Jy=self.com(self.right_hip_position, self.right_knee_position, self.left_hip_position, self.left_knee_position, self.roll)
        pitch_d_1 = 0.0
        pitch_d = -self.pitch_com+pitch_d_1##-0.05817098
        # pitch_d = 0
        theta_pitch = pitch_d - self.pitch
        theta_pitch_pass = self.ST_SIT2_FLC_FM(theta_pitch)#
        # theta_pitch_vel = 0 - self.pitch_dot      

        print(f"self.pitch_d:{pitch_d}")
        print(f"self.pitch:{self.pitch}")
        print(f"self,pitch_com:{self.pitch_com}")

        command = self.command_pitch.compute(error=theta_pitch_pass,dt=0.002)#theta_pitch_pass
        return command



    def get_command_velocity(self):
        self.velocity_d = 0
    
    
        theta_velocity = self.robot_x_velocity - self.velocity_d      
        command = self.command_velocity.compute(error=theta_velocity,dt=0.002)#0.002,0.1


        # print(f"command_velocity:{command}")
        
        return command
    # return command, theta_velocity

    def get_command_yaw(self):
        # yaw_d = 3.14*(4/8)
        yaw_d = -3.1415926535
        theta_yaw = yaw_d - self.yaw
        command = self.command_yaw.compute(error=theta_yaw,dt=0.002)#0.002,0.1
        # print('command:',command)
        return command

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
        return L, m, pitch_com, roll_com,J_z,J_y
    

    def ST_SIT2_FLC_FM(self,error_pass):
        error = error_pass/math.pi*180/15

        # gamma_d_g = 75 * np.pi / 180
        # eth = 0.06  # 80 0.15
        # ld_max, ld_min = 0.9, 0.3
        # a = 1.0
        gamma_d_g = self.fuzzy_gamma_d_g  # 原硬编码：75 * np.pi / 180
        eth = self.fuzzy_eth              # 原硬编码：0.06
        ld_max = self.fuzzy_ld_max        # 原硬编码：0.9
        ld_min = self.fuzzy_ld_min        # 原硬编码：0.3
        a = self.fuzzy_a                  # 原硬编码：0.1



        ld_g = (np.cos(gamma_d_g) - eth) / np.cos(gamma_d_g)
        B0_g, B1_g, B2_g = 0, ld_g * np.sin(gamma_d_g), 1
        C0_g, C1_g, C2_g = 0, ld_g * np.cos(gamma_d_g), 1
        K0_g = (B1_g - B0_g) / (C1_g - C0_g)
        K1_g = (B2_g - B1_g) / (C2_g - C1_g)
        N0_g = (B0_g * C1_g - B1_g * C0_g) / (C1_g - C0_g)
        N1_g = (B1_g * C2_g - B2_g * C1_g) / (C2_g - C1_g)


        if error <= -C2_g:
            gamma_g = B2_g * 70 + 10
        elif -C2_g < error < -C1_g:
            gamma_g = (K1_g * abs(error) + N1_g) * 70 + 10
        elif -C1_g <= error < C1_g:
            gamma_g = (K0_g * abs(error)) * 70 + 10
        elif C1_g <= error < C2_g:
            gamma_g = (K1_g * error + N1_g) * 70 + 10
        else:  # error >= C2_g
            gamma_g = B2_g * 70 + 10

        gamma_g_1 = gamma_g * np.pi / 180  # γ_d 角度
        ld_e = abs(error) / np.cos(gamma_g_1)
        ld_e = max(ld_min, min(ld_e, ld_max))
        ld = ld_e

        B0, B1, B2 = 0, ld * np.sin(gamma_g_1), 1
        C0, C1, C2 = 0, ld * np.cos(gamma_g_1), 1

        m0, m1, m2 = a, 1 - a, a

        K0 = 0.5 * ((B1 - B0 * m0) / (C1 * m0 - C0 + abs(error) * (-m0 + 1)) +
                    (B0 - B1 * m1) / (C0 * m1 - C1 + abs(error) * (-m1 + 1)))
        K1 = 0.5 * ((B2 - B1 * m1) / (C2 * m1 - C1 + abs(error) * (-m1 + 1)) +
                    (B1 - B2 * m2) / (C1 * m2 - C2 + abs(error) * (-m2 + 1)))
        N0 = 0.5 * ((B1 * C0 - B0 * C1 * m0) / (-C1 * m0 + C0 + abs(error) * (m0 - 1)) +
                    (B0 * C1 - B1 * C0 * m1) / (-C0 * m1 + C1 + abs(error) * (m1 - 1)))
        N1 = 0.5 * ((B2 * C1 - B1 * C2 * m1) / (-C2 * m1 + C1 + abs(error) * (m1 - 1)) +
                    (B1 * C2 - B2 * C1 * m2) / (-C1 * m2 + C2 + abs(error) * (m2 - 1)))

        if error <= -C2:
            phi = -B2*15/180*math.pi + (error + C2)
        elif -C2 < error < -C1:
            phi = (K1 * error - N1)*15/180*math.pi
        elif -C1 <= error < C1:
            phi = (K0 * error)*15/180*math.pi
        elif C1 <= error < C2:
            phi = (K1 * error + N1)*15/180*math.pi
        else:  # error >= C2
            phi = B2*15/180*math.pi + (error - C2)

        return phi

    def balance(self):
        a = self.get_command_pitch()
        b = self.get_command_velocity()
        c = self.get_command_yaw()
        left_command = -0.5*a+0.5*b-c
        right_command = 0.5*a-0.5*b-c
        # 通过Genesis接口设置车轮力矩（替代ROS1的command_pub_R3/L3.publish）
        left_command = np.array([left_command], dtype=np.float32)
        right_command = np.array([right_command], dtype=np.float32)
        self.robot.control_dofs_force(left_command, [self.joint_name_to_idx["left_wheel"]])
        self.robot.control_dofs_force(right_command, [self.joint_name_to_idx["right_wheel"]])
        # fuzzy_output_l = torch.tensor(left_command, dtype=torch.float32)
        # fuzzy_output_r = torch.tensor(right_command, dtype=torch.float32)
        # fuzzy_output = torch.cat((fuzzy_output_l, fuzzy_output_r), dim=0)
        # return fuzzy_output
        
class VMC:
    def __init__(self,robot):
        self.dt = 0.002
        self.robot = robot
        self.position_lh=0.0
        self.position_lk=0.0
        self.position_rh=0.0
        self.position_rk=0.0
        self.position_lw=0.0
        self.position_rw=0.0
        self.velocity_lh=0.0
        self.velocity_lw=0.0
        self.velocity_lk=0.0
        self.velocity_rh=0.0
        self.velocity_rk=0.0
        self.velocity_rw=0.0
        self.roll_dot = 0
        self.roll = 0
        self.base_quat = [0,0,0,0]

        self.joint_names = [
            "L1_joint",    # 0: 左髋
            "L2_joint",     # 1: 左膝
            "R1_joint",   # 2: 右髋
            "R2_joint",    # 3: 右膝
            "L3_joint",    # 4: 左轮
            "R3_joint"    # 5: 右轮
        ]
        self.joint_name_to_idx = {
            "left_wheel": 11,    # 对应motor_dofs[4]
            "right_wheel": 10,    # 对应motor_dofs[5]
            "left_hip": 7,       # 对应motor_dofs[0]
            "left_knee": 9,      # 对应motor_dofs[1]
            "right_hip": 6,      # 对应motor_dofs[2]
            "right_knee": 8      # 对应motor_dofs[3]
        }

    #拿到电机反馈关节位置
    def update_joint_states(self):
        # 从消息中提取仿真中的关节速度信息
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        # print(f"self.jopint_names:{self.joint_names}")  # self.jopint_names:['left_thigh_joint', 'left_calf_joint', 'right_thigh_joint', 'right_calf_joint', 'left_wheel_joint', 'right_wheel_joint']
        # print(f"self.motor_dofs:{self.motor_dofs}")  # self.motor_dofs:[7, 9, 6, 8, 11, 10]
        # print(f"self.name:{[self.robot.get_joint(name) for name in self.joint_names]}")

        left_wheel_index = self.motor_dofs[4] # 左轮关节
        right_wheel_index = self.motor_dofs[5] # 右轮关节
        left_hip_index = self.motor_dofs[0]  # 左轮关节
        right_hip_index = self.motor_dofs[2]  # 右轮关节
        left_knee_index = self.motor_dofs[1]  # 左轮关节
        right_knee_index = self.motor_dofs[3]  # 右轮关节

        # print(f"left_wheel_index:{left_wheel_index}")
        # print(f"right_wheel_index:{right_wheel_index}")
        # print(f"left_hip_index:{left_hip_index}")
        # print(f"right_hip_index:{right_hip_index}")
        # print(f"left_knee_index:{left_knee_index}")
        # print(f"right_knee_index:{right_knee_index}")

        # print(f"pos:{self.robot.get_dofs_position()}")
        # ===============useful!!!=============================
        # print(f"left_wheel_index:{left_wheel_index}")
        # print(f"self.position_lw:{self.position_lw}")
        # print(f"self.robot.get_dofs_position():{self.robot.get_dofs_position()}")
        # print(f"self.robot.get_joint:{self.robot.get_joint}")
        # pos = self.robot.get_dofs_position()
        # for name in self.joint_names:
        #     idx = self.robot.get_joint(name).dof_idx_local
        #     print(f"{name}: idx={idx}, pos={pos[idx].item()}")

        self.position_lw = float((self.robot.get_dofs_position()[:,left_wheel_index]))  # 左轮位置（弧度）
        self.position_rw = float((self.robot.get_dofs_position()[:,right_wheel_index]))  # 右轮位置（弧度）
        self.position_lh = float((self.robot.get_dofs_position()[:,left_hip_index])) 
        self.position_rh = float((self.robot.get_dofs_position()[:,right_hip_index]))
        self.position_lk = float((self.robot.get_dofs_position()[:,left_knee_index]))  
        self.position_rk = float((self.robot.get_dofs_position()[:,right_knee_index]))
        ####
        self.velocity_lw = float((self.robot.get_dofs_velocity()[:,left_wheel_index]))  # 左轮速度（弧度/秒）#这里左轮速度是负的，要进行取负处理！
        self.velocity_rw = float((self.robot.get_dofs_velocity()[:,right_wheel_index]))
        self.velocity_lk = float((self.robot.get_dofs_velocity()[:,left_knee_index])) 
        self.velocity_rk = float((self.robot.get_dofs_velocity()[:,right_knee_index]))
        self.velocity_lh = float((self.robot.get_dofs_velocity()[:,left_hip_index]))
        self.velocity_rh = float((self.robot.get_dofs_velocity()[:,right_hip_index]) ) 

    def update_imu_data(self): 
        # 获取base_link角速度信息
        w_x = self.robot.get_ang()[:,0]
        # print(f"w_x:{w_x}")
        # print(f"self.robot.get_vel():{self.robot.get_vel()}")
        # print(f"w_x,w_y,w_z:{w_x},{w_y},{w_z}")

        # 读取机器人imu的四元素
        quat = self.robot.get_quat()
        # print(f"quat1:{self.robot.get_quat()}")
        if isinstance(quat, torch.Tensor):
            quat = quat.detach().cpu().numpy()
        elif isinstance(quat, (list, tuple)):
            quat = np.array(quat, dtype=np.float32)
        # print(f"quat2:{quat}")
        w, x, y, z = quat[:,0] , quat[:,1], quat[:,2], quat[:,3] ### genesis output is [w, x, y, z]!!!

        # 将四元素转化为欧拉角并对实际情况进行数据处理    
        # rpy_angle 中存放的就是欧拉角，分别是绕 x、y、z 轴的角度
        rpy_angle = euler.quat2euler([w, x, y, z])
        self.roll=rpy_angle[0]
        # self.pitch = rpy_angle[1]
        # self.yaw = rpy_angle[2]
        # self.yaw_dot = w_z
        # self.pitch_dot = w_y
        self.roll_dot = float(w_x.cpu().item())


    def ST_SIT2_FLC_FM(self,error_pass):
        error = error_pass/math.pi*180/10
        gamma_d_g = 70 * np.pi / 180
        eth = 0.05  # 80 0.15
        ld_g = (np.cos(gamma_d_g) - eth) / np.cos(gamma_d_g)
        B0_g, B1_g, B2_g = 0, ld_g * np.sin(gamma_d_g), 1
        C0_g, C1_g, C2_g = 0, ld_g * np.cos(gamma_d_g), 1
        K0_g = (B1_g - B0_g) / (C1_g - C0_g)
        K1_g = (B2_g - B1_g) / (C2_g - C1_g)
        N0_g = (B0_g * C1_g - B1_g * C0_g) / (C1_g - C0_g)
        N1_g = (B1_g * C2_g - B2_g * C1_g) / (C2_g - C1_g)
        ld_max, ld_min = 0.75, 0.1

        if error <= -C2_g:
            gamma_g = B2_g * 70 + 10
        elif -C2_g < error < -C1_g:
            gamma_g = (K1_g * abs(error) + N1_g) * 70 + 10
        elif -C1_g <= error < C1_g:
            gamma_g = (K0_g * abs(error)) * 70 + 10
        elif C1_g <= error < C2_g:
            gamma_g = (K1_g * error + N1_g) * 70 + 10
        else:  # error >= C2_g
            gamma_g = B2_g * 70 + 10

        gamma_g_1 = gamma_g * np.pi / 180  # γ_d 角度
        ld_e = abs(error) / np.cos(gamma_g_1)
        ld_e = max(ld_min, min(ld_e, ld_max))
        ld = ld_e

        a = 0.05  # 0.1
        B0, B1, B2 = 0, ld * np.sin(gamma_g_1), 1
        C0, C1, C2 = 0, ld * np.cos(gamma_g_1), 1

        m0, m1, m2 = a, 1 - a, a

        K0 = 0.5 * ((B1 - B0 * m0) / (C1 * m0 - C0 + abs(error) * (-m0 + 1)) +
                    (B0 - B1 * m1) / (C0 * m1 - C1 + abs(error) * (-m1 + 1)))
        K1 = 0.5 * ((B2 - B1 * m1) / (C2 * m1 - C1 + abs(error) * (-m1 + 1)) +
                    (B1 - B2 * m2) / (C1 * m2 - C2 + abs(error) * (-m2 + 1)))
        N0 = 0.5 * ((B1 * C0 - B0 * C1 * m0) / (-C1 * m0 + C0 + abs(error) * (m0 - 1)) +
                    (B0 * C1 - B1 * C0 * m1) / (-C0 * m1 + C1 + abs(error) * (m1 - 1)))
        N1 = 0.5 * ((B2 * C1 - B1 * C2 * m1) / (-C2 * m1 + C1 + abs(error) * (m1 - 1)) +
                    (B1 * C2 - B2 * C1 * m2) / (-C1 * m2 + C2 + abs(error) * (m2 - 1)))

        if error <= -C2:
            phi = -B2*10/180*math.pi + (error + C2)
        elif -C2 < error < -C1:
            phi = (K1 * error - N1)*10/180*math.pi
        elif -C1 <= error < C1:
            phi = (K0 * error)*10/180*math.pi
        elif C1 <= error < C2:
            phi = (K1 * error + N1)*10/180*math.pi
        else:  # error >= C2
            phi = B2*10/180*math.pi + (error - C2)

        return phi        
        

    def vmc(self):
        k=400#120
        d=40
        theta_10=70*math.pi/180
        theta_20=95*math.pi/180
        l1=0.15
        l2=0.25
        x_d=0.0353
        z_d=0.2779
        z_d_l=2500*self.roll+100*self.roll_dot
        x_dot_d=0
        z_dot_d=0
        x_l=l1*math.sin(theta_10-self.position_lh)-l2*math.sin((theta_20+self.position_lk)-(theta_10-self.position_lh))
        z_l=l1*math.cos(theta_10-self.position_lh)+l2*math.cos((theta_20+self.position_lk)-(theta_10-self.position_lh))
        x_r=l1*math.sin(theta_10+self.position_rh)-l2*math.sin((theta_20-self.position_rk)-(theta_10+self.position_rh))
        z_r=l1*math.cos(theta_10+self.position_rh)+l2*math.cos((theta_20-self.position_rk)-(theta_10+self.position_rh))
        jcobi_l=np.matrix([[ -l1*math.cos(theta_10-self.position_lh)-l2*math.cos((theta_20+self.position_lk)-(theta_10-self.position_lh))  ,  -l2*math.cos((theta_20+self.position_lk)-(theta_10-self.position_lh))],
                        [ l1*math.sin(theta_10-self.position_lh)-l2*math.sin((theta_20+self.position_lk)-(theta_10-self.position_lh))  ,   -l2*math.sin((theta_20+self.position_lk)-(theta_10-self.position_lh))]])
        jcobi_r=np.matrix([[ l1*math.cos(theta_10+self.position_rh)+l2*math.cos((theta_20-self.position_rk)-(theta_10+self.position_rh))  ,  l2*math.cos((theta_20-self.position_rk)-(theta_10+self.position_rh))],
                        [ -l1*math.sin(theta_10+self.position_rh)+l2*math.sin((theta_20-self.position_rk)-(theta_10+self.position_rh))  ,   l2*math.sin((theta_20-self.position_rk)-(theta_10+self.position_rh))]])
        theta_dot_l=np.matrix([[self.velocity_lh],[self.velocity_lk]])
        theta_dot_r=np.matrix([[self.velocity_rh],[self.velocity_rk]])
        p_dot_l=jcobi_l*theta_dot_l
        p_dot_r=jcobi_r*theta_dot_r
        # print(p_dot[0][0])
        # F=np.matrix([[k*(x_d-x)+d*(x_dot_d-p_dot[0][0])],[k*(z_d-z)+d*(z_dot_d-p_dot[1][0])]])
        F_l = np.matrix([[k*(x_d-x_l)+d*(x_dot_d-p_dot_l[0, 0])],
               [k*(z_d-z_l)+d*(z_dot_d-p_dot_l[1, 0])+0-z_d_l]])
        F_r = np.matrix([[k*(x_d-x_r)+d*(x_dot_d-p_dot_r[0, 0])],
                    [k*(z_d-z_r)+d*(z_dot_d-p_dot_r[1, 0])+0+z_d_l]])
        # print(F)
        tao_l=jcobi_l.T*F_l
        tao_r=jcobi_r.T*F_r

        self.robot.control_dofs_force([tao_l[0,0]], [self.joint_name_to_idx["left_hip"]])
        self.robot.control_dofs_force([tao_l[1,0]], [self.joint_name_to_idx["left_knee"]])
        self.robot.control_dofs_force([tao_r[0,0]], [self.joint_name_to_idx["right_hip"]])
        self.robot.control_dofs_force([tao_r[1,0]], [self.joint_name_to_idx["right_knee"]])

        # self.robot.control_dofs_force([tao_l[0,0]], [self.joint_name_to_idx["left_hip"]])
        # self.robot.control_dofs_force([tao_l[1,0]], [self.joint_name_to_idx["left_knee"]])
        # self.robot.control_dofs_force([tao_r[0,0]], [self.joint_name_to_idx["right_hip"]])
        # self.robot.control_dofs_force([tao_r[1,0]], [self.joint_name_to_idx["right_knee"]])
        # tao_l= torch.tensor(tao_l, dtype=torch.float32)
        # tao_r= torch.tensor(tao_r, dtype=torch.float32)
        
        # vmc_output_lh = torch.tensor(tao_l[0,0], dtype=torch.float32).unsqueeze(0)
        # vmc_output_lk = torch.tensor(tao_l[1,0], dtype=torch.float32).unsqueeze(0)
        # vmc_output_rh = torch.tensor(tao_r[0,0], dtype=torch.float32).unsqueeze(0)
        # vmc_output_rk = torch.tensor(tao_r[1,0], dtype=torch.float32).unsqueeze(0)
        # vmc_output = torch.cat((vmc_output_lh,vmc_output_lk,vmc_output_rh,vmc_output_rk),0).unsqueeze(0)        

        # print(f"vmc_output_l: {vmc_output_lh}{vmc_output_lk}")
        # print(f"vmc_output_r: {vmc_output_rh}{vmc_output_rk}")

        # print(f"vmc_output: {vmc_output}")
        # return vmc_output
        # return [tao_l[0,0],tao_l[1,0],tao_r[0,0],tao_r[1,0]]



def main():
    gs.init(backend=gs.gpu)
    import torch
    import cv2
    import numpy as np


    from wheel_legged_train_our import get_cfgs  
    env_cfg, _, _, _, _, _, terrain_cfg = get_cfgs()

    dt = 0.002  


    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(0.5 / dt),
            camera_pos=(2.0, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(n_rendered_envs=1),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            batch_dofs_info=True,
        ),
        show_viewer=True,
    )

    # add plane
    scene.add_entity(gs.morphs.URDF(file="assets/terrain/plane/plane.urdf", fixed=True))

    # add terrain (optional, set terrain_cfg["terrain"] = False for flat test)
    if terrain_cfg["terrain"]:
        height_field = cv2.imread("assets/terrain/png/" + terrain_cfg["train"] + ".png", cv2.IMREAD_GRAYSCALE)
        scene.add_entity(gs.morphs.Terrain(
            height_field=height_field,
            horizontal_scale=terrain_cfg["horizontal_scale"],
            vertical_scale=terrain_cfg["vertical_scale"],
        ))

    # add robot —— 完全复用 env 的逻辑
    base_init_pos = np.array(env_cfg["base_init_pos"]["urdf"], dtype=np.float32)
    base_init_quat = np.array(env_cfg["base_init_quat"], dtype=np.float32)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="/home/zhjkoi/wheel_leg_paper/wheel_legged_genesis/wheel_leg/src/description/urdf/wheel_leg.urdf",
            pos=(-1, -1, 0.42),
            quat=base_init_quat,
        ),
    )
    scene.build(n_envs=1)  # 单环境

    # ================ 3. 初始化控制器 ================
    vmc = VMC(robot)
    lqr = LQR_Controller(robot)





    while True:
        vmc.update_joint_states()
        vmc.update_imu_data()
        vmc.vmc()      # → output hip/knee torque
        lqr.update_joint_states()
        lqr.update_imu_data()
        lqr.balance()  # → output wheel torque


        scene.step()
if __name__ == "__main__":
    main()

