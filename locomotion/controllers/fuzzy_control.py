
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
        
        # # 创建对象
        # # 实例化双腿关节位置控制话题的发布者对象
        # self.command_pub_R1 = rospy.Publisher("/wheel_leg/joint1_position_controller/command", Float64, queue_size=10)
        # self.command_pub_R2 = rospy.Publisher("/wheel_leg/joint2_position_controller/command", Float64, queue_size=10)
        # self.command_pub_L1 = rospy.Publisher("/wheel_leg/joint4_position_controller/command", Float64, queue_size=10)
        # self.command_pub_L2 = rospy.Publisher("/wheel_leg/joint5_position_controller/command", Float64, queue_size=10)
        # # 实例化左右轮力矩控制话题的发布者对象
        # self.command_pub_R3 = rospy.Publisher("/wheel_leg/joint3_position_controller/command", Float64, queue_size=10)
        # self.command_pub_L3 = rospy.Publisher("/wheel_leg/joint6_position_controller/command", Float64, queue_size=10) 
        # # 实例化关节状态话题的订阅者对象
        # self.state_sub_duqu    = rospy.Subscriber('/wheel_leg/joint_states', JointState, self.duqu_callback)  # 获取机器人的位置与速度
        # # 订阅imu（为了获取倾斜角度的值）
        # self.theta=rospy.Subscriber("/wheel_leg/imu", Imu, self.imu_data)
        # 初始化参数
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
        # self.command = PIDController(kp=20, ki=0, kd=2)
        # self.command_yaw = PIDController(kp=10, ki=0, kd=0.5)#kp=5, ki=0, kd=0.03
        # self.command_pitch = PIDController(kp=90,ki=0.0,kd=11.5)#kp=25,ki=0.0,kd=0.5
        # self.command_velocity = PIDController(kp=5,ki=0.5,kd=0)#kp=25,ki=0.0,kd=0.5
        # self.command_yaw = PIDController(kp=10, ki=0, kd=1)#kp=5, ki=0, kd=0.03
        # self.command_pitch = PIDController(kp=40,ki=0.0,kd=4)#kp=25,ki=0.0,kd=0.5
        # self.command_velocity = PIDController(kp=5,ki=0.5,kd=0)#kp=25,ki=0.0,kd=0.5
        # self.command_yaw = PIDController(kp=6, ki=0, kd=1)#kp=5, ki=0, kd=0.03
        # self.command_pitch = PIDController(kp=15,ki=0.0,kd=1.5)#kp=25,ki=0.0,kd=0.5
        # self.command_velocity = PIDController(kp=1.5,ki=0.01,kd=0)#kp=25,ki=0.0,kd=0.5
        # self.command_yaw = PIDController(kp=6, ki=0, kd=1)#kp=5, ki=0, kd=0.03
        # self.command_pitch = PIDController(kp=28,ki=0.0,kd=4)#kp=25,ki=0.0,kd=0.5
        # self.command_velocity = PIDController(kp=5,ki=0.01,kd=0)#kp=25,ki=0.0,kd=0.5
        # self.command_yaw = PIDController(kp=6, ki=0, kd=1)#kp=5, ki=0, kd=0.03
        # self.command_pitch = PIDController(kp=25,ki=0.0,kd=2)#kp=25,ki=0.0,kd=0.5
        # self.command_velocity = PIDController(kp=6.5,ki=0.01,kd=0)#kp=25,ki=0.0,kd=0.5
        self.command_yaw = PIDController(kp=0, ki=0, kd=3)#kp=5, ki=0, kd=0.03
        self.command_pitch = PIDController(kp=300, ki=0, kd=0)#435 140    #kp=25,ki=0.0,kd=0.5kp=109.60223629, ki=0, kd=10.61355606
        self.command_velocity = PIDController(kp=0, ki=0/200, kd=0)#kp=25,ki=0.0,kd=0.5
        # self.command_velocity = PIDController(kp=-7.55, ki=-7.55/200, kd=0.05)#kp=25,ki=0.0,kd=0.5


        self.joint_names = [
            "left_thigh_joint",    # 0: 左髋
            "left_calf_joint",     # 1: 左膝
            "right_thigh_joint",   # 2: 右髋
            "right_calf_joint",    # 3: 右膝
            "left_wheel_joint",    # 4: 左轮
            "right_wheel_joint"    # 5: 右轮
        ]
        self.joint_name_to_idx = {
            "left_wheel": 10,    # 对应motor_dofs[4]
            "right_wheel": 11,    # 对应motor_dofs[5]
        # }

        # # 在VMC类的__init__方法中，修改关节索引映射：
        # self.joint_name_to_idx = {
            "left_hip": 6,       # 对应motor_dofs[0]
            "left_knee": 8,      # 对应motor_dofs[1]
            "right_hip": 7,      # 对应motor_dofs[2]
            "right_knee": 9      # 对应motor_dofs[3]
        }
        # self.link_names = [ 
        #     "base_link"
        #     # "left_calf_link",
        #     # "left_thigh_link",
        #     # "left_knee_link",
        #     # "right_calf_link",
        #     # "right_thigh_link",
        #     # "right_knee_link",
        #         ],
    


    def update_joint_states(self):
        # 提取仿真中的关节信息
        # 0-based 局部索引，长度 = 总自由度，单环境安全
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
        # print('===========================================')

        # print(f"self.left_wheel_position:{self.left_wheel_position}")
        # print(f"self.robot.get_dofs_position():{self.robot.get_dofs_position()}")
        # print(f"shape:{self.robot.get_dofs_position().shape}")
        # print(f"self.robot.get_dofs_position()[left_wheel_index]:{self.robot.get_dofs_position()[left_wheel_index]}")
        self.left_wheel_position = float((self.robot.get_dofs_position()[left_wheel_index]))  # 左轮位置（弧度）
        self.right_wheel_position = float((self.robot.get_dofs_position()[right_wheel_index]))  # 右轮位置（弧度）
        self.left_hip_position = float(self.robot.get_dofs_position()[left_hip_index])  
        self.right_hip_position = float(self.robot.get_dofs_position()[right_hip_index]) 
        self.left_knee_position = float(self.robot.get_dofs_position()[left_knee_index])  
        self.right_knee_position = float(self.robot.get_dofs_position()[right_knee_index])
        # print(f"self.left_wheel_position:{self.left_wheel_position}")
        # print(f"self.right_wheel_position:{self.right_wheel_position}")
        # print(f"self.left_hip_position:{self.left_hip_position}")
        # print(f"self.right_hip_position:{self.right_hip_position}")
        # print(f"self.left_knee_position:{self.left_knee_position}")
        # print(f"self.right_knee_position:{self.right_knee_position}")
        # print(self.left_wheel_position,self.right_wheel_position,self.left_hip_position,self.right_hip_position,self.left_knee_position,self.right_knee_position)
        # print('===========================================')
        ####
        self.left_wheel_effort = self.robot.get_dofs_force()[left_wheel_index]  # 左轮位置（弧度）
        self.right_wheel_effort = self.robot.get_dofs_force()[right_wheel_index]  # 右轮位置（弧度）
        self.left_hip_effort = self.robot.get_dofs_force()[left_hip_index]  
        self.right_hip_effort = self.robot.get_dofs_force()[right_hip_index] 
        self.left_knee_effort = self.robot.get_dofs_force()[left_knee_index]  
        self.right_knee_effort = self.robot.get_dofs_force()[right_knee_index] 
        ####
        self.left_wheel_velocity = float(self.robot.get_dofs_velocity()[left_wheel_index])  # 左轮速度（弧度/秒）#这里左轮速度是负的，要进行取负处理！
        self.right_wheel_velocity = float(self.robot.get_dofs_velocity()[right_wheel_index])
        self.left_knee_velocity = float(self.robot.get_dofs_velocity()[left_knee_index]  )
        self.right_knee_velocity = float(self.robot.get_dofs_velocity()[right_knee_index])
        self.left_hip_velocity = float(self.robot.get_dofs_velocity()[left_hip_index])
        self.right_hip_velocity = float(self.robot.get_dofs_velocity()[right_hip_index])  
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
        self.robot_x_velocity = (self.left_wheel_velocity + self.right_wheel_velocity) * 0.1/ 2.0
        # self.robot_x_velocity = float((self.robot.get_vel()[0]).cpu().item())


    def update_imu_data(self):

        w_x = self.robot.get_ang()[0]
        w_y = self.robot.get_ang()[1]
        w_z = self.robot.get_ang()[2]

        # a_x = self.robot.get_acc()[0]
        # a_y = self.robot.get_acc()[1]
        # a_z = self.robot.get_acc()[2]

        quat = self.robot.get_quat()
        if isinstance(quat, torch.Tensor):
            quat = quat.detach().cpu().numpy()
        elif isinstance(quat, (list, tuple)):

            quat = np.array(quat, dtype=np.float32)
 
        w, x, y, z = quat

        # 将四元素转化为欧拉角并对实际情况进行数据处理        # rpy_angle 中存放的就是欧拉角，分别是绕 x、y、z 轴的角度
        rpy_angle = euler.quat2euler([w, x, y, z])
        self.roll=rpy_angle[0]
        self.pitch = rpy_angle[1]
        self.yaw = rpy_angle[2]
        # 获取base_link角速度信息
        self.yaw_dot = w_z
        self.pitch_dot = w_y
        self.roll_dot = w_x
        # self.roll_dot = float(self.roll_dot.cpu().item())

        # print(f"pitch:{self.pitch},yaw:{self.yaw}")

    def get_command_pitch(self):
        L, M, self.pitch_com, roll_com,Jz,Jy=self.com(self.right_hip_position, self.right_knee_position, self.left_hip_position, self.left_knee_position, self.roll)
        pitch_d_1 = 0
        pitch_d = -self.pitch_com+pitch_d_1##-0.05817098
        # pitch_d = 0
        theta_pitch = pitch_d - self.pitch
        theta_pitch_pass = self.ST_SIT2_FLC_FM(theta_pitch)#
        # theta_pitch_vel = 0 - self.pitch_dot      

        command = self.command_pitch.compute(error=theta_pitch_pass,dt=0.002)#theta_pitch_pass
        return command
        # command2 = b.compute(error=theta_pitch_vel,dt=0.002)#0.002,0.1
        # print(theta_pitch)
        # print(command1,self.pitch)


    def get_command_velocity(self):
        self.velocity_d = 0
        theta_velocity = self.robot_x_velocity - self.velocity_d      
        command = self.command_velocity.compute(error=theta_velocity,dt=0.002)#0.002,0.1

        print(f"robot_velocity:{self.robot.get_vel()}")
        # print(f"command_velocity:{command}")
        
        return command
    # return command, theta_velocity

    def get_command_yaw(self):
        # yaw_d = 3.14*(4/8)
        yaw_d = 0
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
        dlr = 286 * 1e-3
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
        gamma_d_g = 75 * np.pi / 180
        eth = 0.06  # 80 0.15
        ld_g = (np.cos(gamma_d_g) - eth) / np.cos(gamma_d_g)
        B0_g, B1_g, B2_g = 0, ld_g * np.sin(gamma_d_g), 1
        C0_g, C1_g, C2_g = 0, ld_g * np.cos(gamma_d_g), 1
        K0_g = (B1_g - B0_g) / (C1_g - C0_g)
        K1_g = (B2_g - B1_g) / (C2_g - C1_g)
        N0_g = (B0_g * C1_g - B1_g * C0_g) / (C1_g - C0_g)
        N1_g = (B1_g * C2_g - B2_g * C1_g) / (C2_g - C1_g)
        ld_max, ld_min = 0.9, 0.3

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

        a = 0.1  # 0.1
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



        # print(f"command:{left_command},{right_command}")
        # print(f"command_shape:{left_command.shape},{right_command.shape}")

        # self.command_pub_R1.publish(right_command)#话题订阅对象与仿真关节对象相反
        # self.command_pub_L1.publish(left_command)#话题订阅对象与仿真关节对象相反
        # self.command_pub_R2.publish(right_command)#话题订阅对象与仿真关节对象相反
        # self.command_pub_L2.publish(left_command)#话题订阅对象与仿真关节对象相反
        # self.command_pub_R3.publish(right_command)#话题订阅对象与仿真关节对象相反
        # self.command_pub_L3.publish(left_command)#话题订阅对象与仿真关节对象相反
        # print(a,b,self.pitch)

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
        # 创建对象
        # 实例化双腿关节力矩控制话题的发布者对象
        # self.command_pub_R1 = rospy.Publisher("/wheel_leg/joint1_position_controller/command", Float64, queue_size=10)
        # self.command_pub_R2 = rospy.Publisher("/wheel_leg/joint2_position_controller/command", Float64, queue_size=10)
        # self.command_pub_L1 = rospy.Publisher("/wheel_leg/joint4_position_controller/command", Float64, queue_size=10)
        # self.command_pub_L2 = rospy.Publisher("/wheel_leg/joint5_position_controller/command", Float64, queue_size=10)
        
        # 实例化关节状态话题的订阅者对象
        # self.state_sub    = rospy.Subscriber('/wheel_leg/joint_states', JointState, self.robot_state_callback)  # 获取机器人的位置与速度
        # self.theta=rospy.Subscriber("/wheel_leg/imu", Imu, self.imu_data)
        self.joint_names = [
            "left_thigh_joint",    # 0: 左髋
            "left_calf_joint",     # 1: 左膝
            "right_thigh_joint",   # 2: 右髋
            "right_calf_joint",    # 3: 右膝
            "left_wheel_joint",    # 4: 左轮
            "right_wheel_joint"    # 5: 右轮
        ]
        self.joint_name_to_idx = {
            "left_wheel": 10,    # 对应motor_dofs[4]
            "right_wheel": 11,    # 对应motor_dofs[5]
            "left_hip": 6,       # 对应motor_dofs[0]
            "left_knee": 8,      # 对应motor_dofs[1]
            "right_hip": 7,      # 对应motor_dofs[2]
            "right_knee": 9      # 对应motor_dofs[3]
        }

    #拿到电机反馈关节位置
    def update_joint_states(self):
        # 从消息中提取仿真中的关节速度信息
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        # print(f"self.jopint_names:{self.joint_names}")  # self.jopint_names:['left_thigh_joint', 'left_calf_joint', 'right_thigh_joint', 'right_calf_joint', 'left_wheel_joint', 'right_wheel_joint']
        # print(f"self.motor_dofs:{self.motor_dofs}")  
        # print(f"self.name:{[self.robot.get_joint(name) for name in self.joint_names]}")

        left_wheel_index = self.motor_dofs[4] # 左轮关节
        right_wheel_index = self.motor_dofs[5] # 右轮关节
        left_hip_index = self.motor_dofs[0]  # 左轮关节
        right_hip_index = self.motor_dofs[2]  # 右轮关节
        left_knee_index = self.motor_dofs[1]  # 左轮关节
        right_knee_index = self.motor_dofs[3]  # 右轮关节

        # ===============useful!!!=============================
        # print(f"left_wheel_index:{left_wheel_index}")
        # print(f"self.position_lw:{self.position_lw}")
        # print(f"self.robot.get_dofs_position():{self.robot.get_dofs_position()}")
        # print(f"self.robot.get_joint:{self.robot.get_joint}")
        # pos = self.robot.get_dofs_position()
        # for name in self.joint_names:
        #     idx = self.robot.get_joint(name).dof_idx_local
        #     print(f"{name}: idx={idx}, pos={pos[idx].item()}")

        self.position_lw = float((self.robot.get_dofs_position()[left_wheel_index]))  # 左轮位置（弧度）
        self.position_rw = float((self.robot.get_dofs_position()[right_wheel_index]))  # 右轮位置（弧度）
        self.position_lh = float((self.robot.get_dofs_position()[left_hip_index])) 
        self.position_rh = float((self.robot.get_dofs_position()[right_hip_index]))
        self.position_lk = float((self.robot.get_dofs_position()[left_knee_index]))  
        self.position_rk = float((self.robot.get_dofs_position()[right_knee_index]))
        ####
        self.velocity_lw = float((self.robot.get_dofs_velocity()[left_wheel_index]))  # 左轮速度（弧度/秒）#这里左轮速度是负的，要进行取负处理！
        self.velocity_rw = float((self.robot.get_dofs_velocity()[right_wheel_index]))
        self.velocity_lk = float((self.robot.get_dofs_velocity()[left_knee_index])) 
        self.velocity_rk = float((self.robot.get_dofs_velocity()[right_knee_index]))
        self.velocity_lh = float((self.robot.get_dofs_velocity()[left_hip_index]))
        self.velocity_rh = float((self.robot.get_dofs_velocity()[right_hip_index]) ) 
        # print(f"self.velocity_lh:{self.velocity_lh},self.velocity_lk:{self.velocity_lk}")  # <class 'float'> <class 'float'>  float float float float (type(self.velocity_lh), type(self.velocity_lk))
        # print(type(self.velocity_lh), type(self.velocity_lk))
        # pos = self.robot.get_dofs_position()
        # print("get_dofs_position shape:", pos.shape)
        # for name in self.joint_names:
        #     j = self.robot.get_jo
        # int(name)
        #     idx = j.dof_idx_local
        #     print(f"{name}: dof_idx_local={idx}, pos_at_idx={pos[idx].item()}")
        # print(f"self.position_lk:{self.position_lk}")
        # print(f"self.position_rk:{self.position_rk}")
        # print(f"self.position_lh:{self.position_lh}")
        # print(f"self.position_rh:{self.position_rh}")
    def update_imu_data(self):
        # 获取base_link角速度信息
        w_x = self.robot.get_ang()[0]
        # print(f"self.robot.get_vel():{self.robot.get_vel()}")
        # print(f"w_x,w_y,w_z:{w_x},{w_y},{w_z}")


        # # 获取base_link的xian加速度信息

        # self.base_quat= self.robot.get_quat()
        # inv_base_quat = inv_quat(self.base_quat)
        # self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        # self.base_lin_acc = (self.robot.get_vel() - self.last_base_lin_vel[:])/ self.dt
        # a_x = self.base_lin_acc[0]
        # a_y = self.base_lin_acc[1]
        # a_z = self.base_lin_acc[2]


    # ========================================
        # # 读取机器人imu的四元素

        quat = self.robot.get_quat()
        # print(f"quat1:{self.robot.get_quat()}")
        if isinstance(quat, torch.Tensor):
            quat = quat.detach().cpu().numpy()
        elif isinstance(quat, (list, tuple)):
            quat = np.array(quat, dtype=np.float32)
        # print(f"quat2:{quat}")
        w, x, y, z = quat ### genesis output is [w, x, y, z]!!!

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
        # d=200
        theta_10=70*math.pi/180
        theta_20=95*math.pi/180
        l1=0.15
        l2=0.25
        x_d=0.0353
        # z_d=0.1
        z_d=0.2779
        # z_d=0.2

        # z_d=0.25
        
        # z_d=0.3
        # z_d=0.35
        # roll_pass=self.ST_SIT2_FLC_FM(self.roll)
        # z_d_l=1000*roll_pass+100*self.roll_dot
        z_d_l=(2500*self.roll+100*self.roll_dot)
        # z_d_l=0

        # z_d_l = float(z_d_l.cpu().item())
        x_dot_d=0
        z_dot_d=0
        x_l=l1*math.sin(theta_10-self.position_lh)-l2*math.sin((theta_20+self.position_lk)-(theta_10-self.position_lh))
        z_l=l1*math.cos(theta_10-self.position_lh)+l2*math.cos((theta_20+self.position_lk)-(theta_10-self.position_lh))
        x_r=l1*math.sin(theta_10+self.position_rh)-l2*math.sin((theta_20-self.position_rk)-(theta_10+self.position_rh))
        z_r=l1*math.cos(theta_10+self.position_rh)+l2*math.cos((theta_20-self.position_rk)-(theta_10+self.position_rh))
        # print(f"x_l:{x_l},z_l:{z_l},x_r:{x_r},z_r:{z_r}")
        # print(f"x_l.type:{type(x_l)},z_l.type:{type(z_l)},x_r.type:{type(x_r)},z_r.type:{type(z_r)}")


        jcobi_l=np.matrix([[ -l1*math.cos(theta_10-self.position_lh)-l2*math.cos((theta_20+self.position_lk)-(theta_10-self.position_lh))  ,  -l2*math.cos((theta_20+self.position_lk)-(theta_10-self.position_lh))],
                        [ l1*math.sin(theta_10-self.position_lh)-l2*math.sin((theta_20+self.position_lk)-(theta_10-self.position_lh))  ,   -l2*math.sin((theta_20+self.position_lk)-(theta_10-self.position_lh))]])
        jcobi_r=np.matrix([[ l1*math.cos(theta_10+self.position_rh)+l2*math.cos((theta_20-self.position_rk)-(theta_10+self.position_rh))  ,  l2*math.cos((theta_20-self.position_rk)-(theta_10+self.position_rh))],
                        [ -l1*math.sin(theta_10+self.position_rh)+l2*math.sin((theta_20-self.position_rk)-(theta_10+self.position_rh))  ,   l2*math.sin((theta_20-self.position_rk)-(theta_10+self.position_rh))]])

        theta_dot_l=np.matrix([[self.velocity_lh],[self.velocity_lk]])
        theta_dot_r=np.matrix([[self.velocity_rh],[self.velocity_rk]])
        # print("jcobi_l:\n", jcobi_l)
        # print("jcobi_r:\n", jcobi_r)

        # p_dot_l=jcobi_l*theta_dot_l
        # p_dot_r=jcobi_r*theta_dot_r
        p_dot_l=jcobi_l*theta_dot_l
        p_dot_r=jcobi_r*theta_dot_r

        # print(f"jcobi_l: {jcobi_l}",f"type:{type(jcobi_l)}",f"shape:{jcobi_l.shape}")
        # print(f"theta_dot_l: {theta_dot_l}",f"type:{type(theta_dot_l)}",f"shape:{theta_dot_l.shape}")
        # print(f"p_dot_l: {p_dot_l}",f"type:{type(p_dot_l)}",f"shape:{p_dot_l.shape}")
        # print(f"p_dot_l[0,0]",f"type:{type(p_dot_l[0,0])}",f"shape:{p_dot_l[0,0].shape}")
        # # print(p_dot[0][0])
        # print(f"x_d type: {type(x_d)}, x_dot_d type: {type(x_dot_d)}")
        # print(f"k type: {type(k)}, d type: {type(k)}")

        F_l = np.matrix([[k*(x_d - x_l) + d*(x_dot_d - p_dot_l[0,0])],
                [k*(z_d - z_l) + d*(z_dot_d - p_dot_l[1,0])+0-z_d_l]])

        F_r = np.matrix([[k*(x_d - x_r) + d*(x_dot_d - p_dot_r[0,0])],
                        [k*(z_d - z_r) + d*(z_dot_d - p_dot_r[1,0])+0+z_d_l]])

        tao_l= jcobi_l.T*F_l
        tao_r= jcobi_r.T*F_r
        # print(f"tao_l:{tao_l}")
        # print(f"tao_r:{tao_r}")
        self.robot.control_dofs_force([tao_l[0,0]], [self.joint_name_to_idx["left_hip"]])
        self.robot.control_dofs_force([tao_l[1,0]], [self.joint_name_to_idx["left_knee"]])
        self.robot.control_dofs_force([tao_r[0,0]], [self.joint_name_to_idx["right_hip"]])
        self.robot.control_dofs_force([tao_r[1,0]], [self.joint_name_to_idx["right_knee"]])

        print(f"force_:{self.robot.get_dofs_force()}")


        # print(f"left_hip:{tao_l[0,0]}")
        # print(f"left_hip_shape:{tao_l.shape}")
    #         with open('data_l_duo_1.txt', 'a') as file:
    #                 file.write(f"{self.n}, {self.theta_l1}, {self.theta_lb}, {self.theta_l2}, {self.theta_l3}, {self.tao_lw},{self.tao_lk},{self.tao_lh},{self.force_l_x},{self.force_l_y},{self.pitch},{self.roll},{self.theta_l1_dot},{self.theta_lb_dot},{self.theta_l2_dot},{self.theta_l3_dot},{self.theta_l1_dot_dot},{self.theta_lb_dot_dot},{self.theta_l2_dot_dot},{self.theta_l3_dot_dot}\n")
    #         with open('data_r_duo_1.txt', 'a') as file:
    #                 file.write(f"{self.n}, {self.theta_r1}, {self.theta_rb}, {self.theta_r2}, {self.theta_r3}, {self.tao_rw},{self.tao_rk},{self.tao_rh},{self.force_r_x},{self.force_r_y},{self.theta_r1_dot},{self.theta_rb_dot},{self.theta_r2_dot},{self.theta_r3_dot},{self.theta_r1_dot_dot},{self.theta_rb_dot_dot},{self.theta_r2_dot_dot},{self.theta_r3_dot_dot}\n")
    #                 print(self.kaitou)


        # print(f"z_d_l:{z_d_l}")
        
        # self.command_pub_L1.publish(tao_l[0])
        # self.command_pub_L2.publish(tao_l[1])
        # self.command_pub_R1.publish(tao_r[0])
        # self.command_pub_R2.publish(tao_r[1])


def main():
    gs.init(backend=gs.gpu)

    import matplotlib.pyplot as plt
    import time
    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options = gs.options.SimOptions(
            dt = 0.002,
        ),
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (0, -3.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 240,
        ),
        show_viewer = True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )    
    WL = scene.add_entity(
        gs.morphs.URDF(
            file  = '/home/huang/下载/wheel_leg/src/description/meshes/wheel_leg.urdf',
            pos   = (0, 0, 0.38),
            quat = (1.0, 0.0, 0.0, 0.0),
        ),
    )

    scene.build()


    # 初始化控制器
    vmc_controller = VMC(WL)
    lqr_controller = LQR_Controller(WL)    


    while True:
    # for i in range(30):
        # 更新传感器数据
        vmc_controller.update_joint_states()
        vmc_controller.update_imu_data()
        vmc_controller.vmc()

        lqr_controller.update_joint_states()
        lqr_controller.update_imu_data()
        

        
        lqr_controller.balance()
        scene.step() 


        # print(lqr_controller.pitch,lqr_controller.pitch_dot)
        # now = time.time()
        # if now - last_print_time >= PRINT_INTERVAL_SEC:
        #     print(f"force: {lqr_controller.left_wheel_effort}, {lqr_controller.right_wheel_effort}")
        #     last_print_time = now
        # else :
        #     pass
                # print(f"pitch: {lqr_controller.pitch}, pitch_dot: {lqr_controller.pitch_dot}")
            # print(f"force: {lqr_controller.left_wheel_effort}, {lqr_controller.right_wheel_effort}")
        # while not rospy.is_shutdown():
            # 将膝髋关节的位置设置为0
            # b.command_pub_L1.publish(0)
            # b.command_pub_L2.publish(0)
            # b.command_pub_R1.publish(0)
            # b.command_pub_R2.publish(0)
        # a.vmc()
        # b.balance()
        # print(b.pitch,b.pitch_dot,b.yaw,b.yaw_dot,b.roll,b.roll_dot,b.robot_x_position,b.robot_x_velocity,-b.pitch_com,b.velocity_d)
        # b.rate.sleep()

if __name__ == '__main__':
    main()