#!/usr/bin/env python3

# import rospy
import math
from transforms3d import euler
import numpy as np

#PID控制器
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
    def __init__(self):
        # rospy.init_node('LQR_Controller')
        # self.rate = rospy.Rate(500) # 节点执行的频率，单位赫兹
        # 初始化变量来存储左轮和右轮的速度和位置
        self.left_wheel_velocity = 0.0 #左轮
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
        # 创建对象
        # 实例化双腿关节位置控制话题的发布者对象
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
        self.command_yaw = PIDController(kp=6, ki=0, kd=1)#kp=5, ki=0, kd=0.03
        self.command_pitch = PIDController(kp=20, ki=0, kd=1.5)#kp=25,ki=0.0,kd=0.5kp=109.60223629, ki=0, kd=10.61355606
        # self.command_velocity = PIDController(kp=1.5, ki=0.01, kd=0)#kp=25,ki=0.0,kd=0.5
        self.command_velocity = PIDController(kp=2, ki=0.01, kd=0)#kp=25,ki=0.0,kd=0.5

    def duqu_callback(self, data):
        # 从消息中提取仿真中的关节速度信息
        left_wheel_index = data.name.index("L3_joint")  # 左轮关节
        right_wheel_index = data.name.index("R3_joint")  # 右轮关节
        left_hip_index = data.name.index("L1_joint")  # 左轮关节
        right_hip_index = data.name.index("R1_joint")  # 右轮关节
        left_knee_index = data.name.index("L2_joint")  # 左轮关节
        right_knee_index = data.name.index("R2_joint")  # 右轮关节
        self.left_wheel_position = data.position[left_wheel_index]  # 左轮位置（弧度）
        self.right_wheel_position = data.position[right_wheel_index]  # 右轮位置（弧度）
        self.left_hip_position = data.position[left_hip_index]  
        self.right_hip_position = data.position[right_hip_index] 
        self.left_knee_position = data.position[left_knee_index]  
        self.right_knee_position = data.position[right_knee_index] 
        ####
        self.left_wheel_effort = data.effort[left_wheel_index]  # 左轮位置（弧度）
        self.right_wheel_effort = data.effort[right_wheel_index]  # 右轮位置（弧度）
        self.left_hip_effort = data.effort[left_hip_index]  
        self.right_hip_effort = data.effort[right_hip_index] 
        self.left_knee_effort = data.effort[left_knee_index]  
        self.right_knee_effort = data.effort[right_knee_index] 
        ####
        self.left_wheel_velocity = data.velocity[left_wheel_index]  # 左轮速度（弧度/秒）#这里左轮速度是负的，要进行取负处理！
        self.right_wheel_velocity = data.velocity[right_wheel_index]
        self.left_knee_velocity = data.velocity[left_knee_index]  
        self.right_knee_velocity = data.velocity[right_knee_index]
        self.left_hip_velocity = data.velocity[left_hip_index]
        self.right_hip_velocity = data.velocity[right_hip_index]  
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
        self.robot_x_velocity = (self.left_wheel_velocity - self.right_wheel_velocity) * 0.1 / 2.0

    def imu_data(self, imu_data):
        # 下面定义了机器人在base_link上的imu相关接口，可直接调用
        # 读取机器人imu的角速度
        w_x = imu_data.angular_velocity.x
        w_y = imu_data.angular_velocity.y
        w_z = imu_data.angular_velocity.z

        # 读取机器人imu的线加速度
        a_x = imu_data.linear_acceleration.x
        a_y = imu_data.linear_acceleration.y
        a_z = imu_data.linear_acceleration.z
        
        # 读取机器人imu的四元素
        x = imu_data.orientation.x
        y = imu_data.orientation.y
        z = imu_data.orientation.z
        w = imu_data.orientation.w

        # 将四元素转化为欧拉角并对实际情况进行数据处理        # rpy_angle 中存放的就是欧拉角，分别是绕 x、y、z 轴的角度
        rpy_angle = euler.quat2euler([w, x, y, z])
        self.roll=rpy_angle[0]
        self.pitch = rpy_angle[1]
        self.yaw = rpy_angle[2]
        self.yaw_dot = w_z
        self.pitch_dot = w_y
        self.roll_dot = w_x

    def get_command_pitch(self):
        L, M, self.pitch_com, roll_com,Jz,Jy=self.com(self.right_hip_position, self.right_knee_position, self.left_hip_position, self.left_knee_position, self.roll)
        pitch_d_1 = 0
        pitch_d = -self.pitch_com+pitch_d_1##-0.05817098
        theta_pitch = pitch_d - self.pitch
        theta_pitch_pass = self.ST_SIT2_FLC_FM(theta_pitch)
        # theta_pitch_vel = 0 - self.pitch_dot      
        # b = PIDController(kp=0.8,ki=0.0,kd=0.0)
        command = self.command_pitch.compute(error=theta_pitch_pass,dt=0.002)#0.001,0.1
        # command2 = b.compute(error=theta_pitch_vel,dt=0.002)#0.001,0.1
        # print(theta_pitch)
        # print(command1,self.pitch)
        return command

    def get_command_velocity(self):
        self.velocity_d = 2 
        theta_velocity = self.robot_x_velocity - self.velocity_d      
        command = self.command_velocity.compute(error=theta_velocity,dt=0.002)#0.001,0.1
        # print(self.robot_x_velocity,command)
        return command

    def get_command_yaw(self):
        # yaw_d = 3.14*(4/8)
        yaw_d = 0
        theta_yaw = yaw_d - self.yaw
        command = self.command_yaw.compute(error=theta_yaw,dt=0.002)#0.001,0.1
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
        dr1 = np.array([0.075, 0.0, 0.010])
        dr2 = np.array([0.125, 0.0, 0.010])
        dl1 = np.array([0.075, 0.0, -0.010])
        dl2 = np.array([0.125, 0.0, -0.010])
        db = np.array([0, 0, 0])
        # 质量
        mr1 = 1.30110728
        mr2 = 2.00310728
        ml1 = 1.30110728
        ml2 = 2.00310728
        mb = 4.63282560
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
        # self.command_pub_R3.publish(right_command)#话题订阅对象与仿真关节对象相反
        # self.command_pub_L3.publish(left_command)#话题订阅对象与仿真关节对象相反
        # print(a,b,self.pitch)

class VMC:
    def __init__(self):
            
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
        # 创建对象
        # 实例化双腿关节力矩控制话题的发布者对象
        # self.command_pub_R1 = rospy.Publisher("/wheel_leg/joint1_position_controller/command", Float64, queue_size=10)
        # self.command_pub_R2 = rospy.Publisher("/wheel_leg/joint2_position_controller/command", Float64, queue_size=10)
        # self.command_pub_L1 = rospy.Publisher("/wheel_leg/joint4_position_controller/command", Float64, queue_size=10)
        # self.command_pub_L2 = rospy.Publisher("/wheel_leg/joint5_position_controller/command", Float64, queue_size=10)
        
        # 实例化关节状态话题的订阅者对象
        # self.state_sub    = rospy.Subscriber('/wheel_leg/joint_states', JointState, self.robot_state_callback)  # 获取机器人的位置与速度
        # self.theta=rospy.Subscriber("/wheel_leg/imu", Imu, self.imu_data)
    #拿到电机反馈关节位置
    def robot_state_callback(self, data):
        # 从消息中提取左轮和右轮的速度信息
        left_wheel_index = data.name.index("L3_joint")  # 左轮关节
        right_wheel_index = data.name.index("R3_joint")  # 右轮关节
        left_knee_index = data.name.index("L2_joint")  # 左膝关节
        right_knee_index = data.name.index("R2_joint")  # 右膝关节
        left_hip_index = data.name.index("L1_joint")  # 左髋关节
        right_hip_index = data.name.index("R1_joint")  # 右髋关节
        self.position_lw = data.position[left_wheel_index]  # 左轮位置（弧度）
        self.position_rw = data.position[right_wheel_index]  # 右轮位置（弧度）
        self.position_lk = data.position[left_knee_index]  # 左膝位置（弧度）
        self.position_rk = data.position[right_knee_index]  # 右膝位置（弧度）
        self.position_lh = data.position[left_hip_index]  # 左髋位置（弧度）
        self.position_rh = data.position[right_hip_index]  # 右髋位置（弧度）
        self.velocity_lw = data.velocity[left_wheel_index]  # 左轮速度（弧度/秒）
        self.velocity_rw = data.velocity[right_wheel_index]  # 右轮速度（弧度/秒）
        self.velocity_lk = data.velocity[left_knee_index]  # 左膝速度（弧度/秒）
        self.velocity_rk = data.velocity[right_knee_index]  # 右膝速度（弧度/秒）
        self.velocity_lh = data.velocity[left_hip_index]  # 左髋速度（弧度/秒）
        self.velocity_rh = data.velocity[right_hip_index]  # 右髋速度（弧度/秒）
        # print(self.position_lh)

    def imu_data(self, imu_data):
        # 下面定义了机器人在base_link上的imu相关接口，可直接调用
        # 读取机器人imu的角速度
        w_x = imu_data.angular_velocity.x
        w_y = imu_data.angular_velocity.y
        w_z = imu_data.angular_velocity.z

        # 读取机器人imu的线加速度
        a_x = imu_data.linear_acceleration.x
        a_y = imu_data.linear_acceleration.y
        a_z = imu_data.linear_acceleration.z
        
        # 读取机器人imu的四元素
        x = imu_data.orientation.x
        y = imu_data.orientation.y
        z = imu_data.orientation.z
        w = imu_data.orientation.w

        # 将四元素转化为欧拉角并对实际情况进行数据处理        # rpy_angle 中存放的就是欧拉角，分别是绕 x、y、z 轴的角度
        rpy_angle = euler.quat2euler([w, x, y, z])
        self.roll=rpy_angle[0]
        self.roll_dot=w_x

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
        k=800#120
        d=2*k**0.5
        # d=200
        theta_10=70*math.pi/180
        theta_20=95*math.pi/180
        l1=0.15
        l2=0.25
        x_d=0.0353
        # z_d=0.1
        # z_d=0.2779
        # z_d=0.2
        z_d=0.25
        # z_d=0.3
        # z_d=0.35
        # roll_pass=self.ST_SIT2_FLC_FM(self.roll)
        # z_d_l=1000*roll_pass+100*self.roll_dot
        z_d_l=1000*self.roll+100*self.roll_dot
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
        # F_l = np.matrix([[k*(x_d-x_l)+d*(x_dot_d-p_dot_l[0, 0])],
        #        [k*(z_d-z_l)+d*(z_dot_d-p_dot_l[1, 0])+0]])
        F_r = np.matrix([[k*(x_d-x_r)+d*(x_dot_d-p_dot_r[0, 0])],
                    [k*(z_d-z_r)+d*(z_dot_d-p_dot_r[1, 0])+0+z_d_l]])
        # F_r = np.matrix([[k*(x_d-x_r)+d*(x_dot_d-p_dot_r[0, 0])],
        #             [k*(z_d-z_r)+d*(z_dot_d-p_dot_r[1, 0])+0]])
        # print(F)
        tao_l=jcobi_l.T*F_l
        tao_r=jcobi_r.T*F_r
        self.command_pub_L1.publish(tao_l[0])
        self.command_pub_L2.publish(tao_l[1])
        self.command_pub_R1.publish(tao_r[0])
        self.command_pub_R2.publish(tao_r[1])
        # self.command_pub_L1.publish(0)
        # self.command_pub_L2.publish(0)
        # self.command_pub_R1.publish(0)
        # self.command_pub_R2.publish(0)

    # def data(self):
    #     self.kaitou += 1
    #     if self.kaitou >= 1000:
    #         with open('data_l_duo_1.txt', 'a') as file:
    #                 file.write(f"{self.n}, {self.theta_l1}, {self.theta_lb}, {self.theta_l2}, {self.theta_l3}, {self.tao_lw},{self.tao_lk},{self.tao_lh},{self.force_l_x},{self.force_l_y},{self.pitch},{self.roll},{self.theta_l1_dot},{self.theta_lb_dot},{self.theta_l2_dot},{self.theta_l3_dot},{self.theta_l1_dot_dot},{self.theta_lb_dot_dot},{self.theta_l2_dot_dot},{self.theta_l3_dot_dot}\n")
    #         with open('data_r_duo_1.txt', 'a') as file:
    #                 file.write(f"{self.n}, {self.theta_r1}, {self.theta_rb}, {self.theta_r2}, {self.theta_r3}, {self.tao_rw},{self.tao_rk},{self.tao_rh},{self.force_r_x},{self.force_r_y},{self.theta_r1_dot},{self.theta_rb_dot},{self.theta_r2_dot},{self.theta_r3_dot},{self.theta_r1_dot_dot},{self.theta_rb_dot_dot},{self.theta_r2_dot_dot},{self.theta_r3_dot_dot}\n")
    #                 print(self.kaitou)

# def main():
#     a = VMC()
#     b = LQR_Controller()
# # while not rospy.is_shutdown():
#     # 将膝髋关节的位置设置为0
#     # b.command_pub_L1.publish(0)
#     # b.command_pub_L2.publish(0)
#     # b.command_pub_R1.publish(0)
#     # b.command_pub_R2.publish(0)
#     a.vmc()
#     b.balance()
#     print(b.pitch,b.pitch_dot,b.yaw,b.yaw_dot,b.roll,b.roll_dot,b.robot_x_position,b.robot_x_velocity,-b.pitch_com,b.velocity_d)
    # b.rate.sleep()

# if __name__ == '__main__':
#     main()