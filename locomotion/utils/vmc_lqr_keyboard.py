#!/usr/bin/env python3
from control_msgs.msg import JointControllerState
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
import rospy
import control
import math
import numpy as np
from nav_msgs.msg import Odometry
from transforms3d import euler
from rosgraph_msgs.msg import  Clock
import torch

wheel_radius=0.1
# # 计算出K
# K = np.matrix([[ -1.   ,  -1.86370082 ,-11.04295304  ,-2.3038782  ,  0.70710678 ,  0.75591345],
#                 [ -1.   ,  -1.86370082 ,-11.04295304  ,-2.3038782 ,  -0.70710678 , -0.75591345]])

#左轮lqr控制
class LQR_Controller:
    def __init__(self):
        rospy.init_node('LQR_Controller')
        self.rate = rospy.Rate(500) # 节点执行的频率，单位赫兹

        
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
        self.roll=0.0
        self.pitch=0.0
        self.yaw=0.0
        self.roll_dot=0.0
        self.pitch_dot=0.0
        self.yaw_dot=0.0
        self.pitch_com=0.0
        self.slope_tra=0
        self.ax=0
        self.ay=0
        self.az=0
        self.velocity=0
        self.trip_count = 0
        self.prev_segment =0
        self.roll_dot_dot  =0
        self.pitch_dot_dot =0
        self.yaw_dot_dot   =0
        self.current_position_x =  0
        self.torque_lw_pred=0
        self.torque_rw_pred=0
        self.wheel_position_x=0
                
        self.x=0
        self.y=0
        self.z=0
        

        self.vx=0
        self.vy=0
        self.vz=0
        # 创建对象
        # 实例化左右轮力矩控制话题的发布者对象
        self.command_pub_R3 = rospy.Publisher("/wheel_leg/joint3_position_controller/command", Float64, queue_size=10)
        self.command_pub_L3 = rospy.Publisher("/wheel_leg/joint6_position_controller/command", Float64, queue_size=10) 
        # 实例化关节状态话题的订阅者对象
        self.state_sub    = rospy.Subscriber('/wheel_leg/joint_states', JointState, self.robot_state_callback)  # 获取机器人的位置与速度
        # 订阅imu（为了获取倾斜角度的值）
        self.theta=rospy.Subscriber("/wheel_leg/imu", Imu, self.imu_data)
        # 订阅里程计(获取机器人的位置与速度)
        # self.position_and_velocity_sub = rospy.Subscriber('/wheel_leg/odom', Odometry, self.odom_callback)
        #键盘速度与转角
        self.speed_sub=rospy.Subscriber("/keyboard_v_pub", Float64, self.speed_d)
        self.rotaion_sub=rospy.Subscriber("/keyboard_r_pub", Float64, self.rotation_d)
        rospy.Subscriber("/clock", Clock, self.clock)
        # 订阅第一个里程计（L3_link）
        self.position_and_velocity_sub = rospy.Subscriber('/wheel_leg/odom', Odometry, self.odom_callback)
        # 订阅第二个里程计（base_link）
        self.base_position_sub = rospy.Subscriber('/wheel_leg/odom_base', Odometry, self.base_odom_callback)




        # 说明：LQR控制需要参数：theta、theta_dot、y、y_dot,将这四个值放在向量current_state中
        # 初始化参数
        self.speed=0.0
        self.rotation=0.0
        self.flag =True
        self.current_state = np.matrix([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])# 机器人状态变量X的当前值，初始化
        self.desired_state = np.matrix([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]) #机器人状态变量X的期望值为一乘6的矩阵
        self.command_msg_L = Float64()#左轮lqr控制率
        self.command_msg_R = Float64()#右轮lqr控制率
        self.sim_time = 0.0
        self.position_x=0

    def clock(self,clock_data):
        self.sim_time=clock_data.clock.secs+clock_data.clock.nsecs*1e-9
        # cycle_duration = 16 # 每个来回周期总时长10秒
        # half_cycle = cycle_duration / 2  # 每一半周期5秒
        # total_cycles = 8 # 总共来回15次
        # total_time = total_cycles * cycle_duration

        # if self.sim_time < total_time:
        #     current_cycle_time = self.sim_time % cycle_duration
        #     if current_cycle_time < half_cycle:
        self.desired_state[1][0] = 1.5
        #     else:
        #         self.desired_state[1][0] = -1.0
        # else:
        #     self.desired_state[1][0] = 0.0  # 超出时间后可以保持为0或任意值

    def odom_callback(self,msg):
        # 处理里程计消息的回调函数
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        linear_velocity = msg.twist.twist.linear
        angular_velocity = msg.twist.twist.angular
        x=position.x
        y=position.y
        z=position.z
        self.slope_tra=z
        self.wheel_position_x = x
        
    def base_odom_callback(self, msg):
        """
        处理基座 base_link 里程计消息的回调函数

        参数:
            msg (nav_msgs.msg.Odometry): 里程计消息

        功能:
            - 读取基座当前位置和姿态
            - 读取线速度和角速度
            - 更新类成员变量（你需要用的变量）
        """
        # 读取位置
        position = msg.pose.pose.position
        x = position.x
        y = position.y
        z = position.z

        # 读取姿态四元数
        orientation = msg.pose.pose.orientation
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w

        # 读取线速度
        linear_velocity = msg.twist.twist.linear
        vx = linear_velocity.x
        vy = linear_velocity.y
        vz = linear_velocity.z
        
        self.x=x
        self.y=y
        self.z=z


        self.vx=vx
        self.vy=vy
        self.vz=vz


    def speed_d(self,data):
        self.speed=data.data
        self.desired_state[1]=self.speed
        
    def rotation_d(self,data):
        self.rotation=data.data
        self.desired_state[4]=self.rotation



    def robot_state_callback(self, data):
        # 从消息中提取左轮和右轮的速度信息
        left_wheel_index = data.name.index("L3_joint")  # 左轮关节
        right_wheel_index = data.name.index("R3_joint")  # 右轮关节
        left_hip_index = data.name.index("L1_joint")  # 左髋关节
        right_hip_index = data.name.index("R1_joint")  # 右髋关节
        left_knee_index = data.name.index("L2_joint")  # 左膝关节
        right_knee_index = data.name.index("R2_joint")  # 右膝关节

        self.left_wheel_position = data.position[left_wheel_index]  # 左轮位置（弧度）
        self.right_wheel_position = -data.position[right_wheel_index]  # 右轮位置（弧度）
        self.left_hip_position = data.position[left_hip_index]  
        self.right_hip_position = data.position[right_hip_index] 
        self.left_knee_position = data.position[left_knee_index]  
        self.right_knee_position = data.position[right_knee_index] 


        # 获取当前轮子位置和速度
        current_left_pos = data.position[left_wheel_index]
        current_right_pos = -data.position[right_wheel_index]  # 右轮取反（如果方向相反）
        current_left_vel = data.velocity[left_wheel_index]
        current_right_vel = -data.velocity[right_wheel_index]  # 右轮速度取反（如果方向相反）

        # 初始化前一次位置（如果是第一次运行）
        if not hasattr(self, 'prev_left_pos'):
            self.prev_left_pos = current_left_pos
            self.prev_right_pos = current_right_pos
            self.total_distance = 0.0  # 总路程（累加绝对值）
            self.current_x = 0.0  # 当前位置（可正可负）

        # 计算两轮位移变化（考虑角度回绕）
        delta_left = current_left_pos - self.prev_left_pos
        delta_right = current_right_pos - self.prev_right_pos

        # 更新前一次位置
        self.prev_left_pos = current_left_pos
        self.prev_right_pos = current_right_pos

        # 计算x方向位移（平均两轮变化）
        delta_x = (delta_left + delta_right) * wheel_radius / 2.0

        # 更新总路程（绝对值累加）
        self.total_distance += abs(delta_x)  # 关键：用绝对值累加，确保来回都算路程

        # 更新当前位置（可正可负）
        self.current_x += delta_x

        # 计算当前速度
        current_velocity = (current_left_vel + current_right_vel) * wheel_radius / 2.0

        # 更新状态（根据你的控制逻辑）
        if abs(self.desired_state[1]) < 0.01:  # 当期望速度很小时
            self.current_state[0] = self.current_x  # 或者用 self.total_distance 取决于你的需求
            if self.flag:
                self.desired_state[0] = self.current_x
                self.flag = False
        else:
            self.current_state[0] = 0
            self.desired_state[0] = 0
            self.flag = True

        self.current_state[1] = current_velocity
        
        self.position_x = self.total_distance # 或者 self.total_distance 取决于你的需求
        
        self.current_position_x =  self.current_x # 或者 self.total_distance 取决于你的需求

        self.velocity = current_velocity



        # single_way_distance = 10.0  # 单程距离：7米
        # total_round_trips = 8  # 总共往返8次
        # total_distance = 2 * single_way_distance * total_round_trips  # 总运动距离 = (7m去 + 7m回) × 8次

        # if self.total_distance < total_distance:
        #     current_leg_distance = self.total_distance % (2 * single_way_distance)  # 当前往返段内的距离
        #     if current_leg_distance < single_way_distance:
        #         self.desired_state[1][0] = 1.0  # 正向运动（去）
        #     else:
        #         self.desired_state[1][0] = -1.0  # 反向运动（回）
        # else:
        #     self.desired_state[1][0] = 0.0  # 运动结束，停止
 




    def imu_data(self, imu_data):
        # 下面定义了机器人在base_link上的imu相关接口，可直接调用
        # 读取机器人imu的角速度
        w_x = imu_data.angular_velocity.x
        w_y = imu_data.angular_velocity.y
        w_z = imu_data.angular_velocity.z

    # 初始化角加速度计算（如果是第一次调用）
        if not hasattr(self, 'last_imu_time'):
            self.last_imu_time = rospy.Time.now().to_sec()
            self.last_w_x = w_x
            self.last_w_y = w_y
            self.last_w_z = w_z
            self.w_x_dot = 0.0
            self.w_y_dot = 0.0
            self.w_z_dot = 0.0
            return  # 跳过第一次计算（因为没有历史数据）

        # 计算时间差（避免除以零）
        current_time = rospy.Time.now().to_sec()
        dt = current_time - self.last_imu_time
        if dt <= 0:
            return  # 时间差无效，跳过本次计算

        # 数值微分计算角加速度
        self.w_x_dot = (w_x - self.last_w_x) / dt  # roll 角加速度
        self.w_y_dot = (w_y - self.last_w_y) / dt  # pitch 角加速度
        self.w_z_dot = (w_z - self.last_w_z) / dt  # yaw 角加速度

        # 更新历史数据
        self.last_imu_time = current_time
        self.last_w_x = w_x
        self.last_w_y = w_y
        self.last_w_z = w_z


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
        self.pitch=rpy_angle[1]
        self.yaw=rpy_angle[2]
        self.roll_dot=w_x
        self.pitch_dot=w_y
        self.yaw_dot=w_z

        self.roll_dot_dot  =self.w_x_dot
        self.pitch_dot_dot =self.w_y_dot
        self.yaw_dot_dot   =self.w_z_dot

        self.ax=a_x
        self.ay=a_y
        self.az=a_z





        self.current_state[2] =  rpy_angle[1]+self.pitch_com
        self.current_state[3] =  w_y
        self.current_state[4] = rpy_angle[2]#建模相反
        self.current_state[5] =  w_z

        # self.x=x
        # self.y=y
        # self.z=z

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

    def K_cal(self,L,M,Jz,Jy):
        # # 机器人的参数
        m = 1.3401 #轮子质量
        g = 9.8
        wheel_radius = 0.1
        I = 0.0062845  #驱动轮绕轮轴的转动惯量
        D = 0.355 #左右驱动轮之间的距离

        # 状态空间方程中的A、B矩阵
        A = np.matrix([ [0, 1, 0, 0, 0, 0],
                        [0, 0, (-1*M*wheel_radius*L*M*g*L)/(wheel_radius*(M+2*m+(2*I/wheel_radius**2))*(Jz+M*L**2)-M*wheel_radius*L*M*L), 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0,((wheel_radius*(M+2*m+(2*I/wheel_radius**2))*M*g*L)/(wheel_radius*(M+2*m+(2*I/wheel_radius**2))*(Jz+M*L**2)-M*wheel_radius*L*M*L)), 0, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]
                        ])

        B = np.matrix([ [0, 0],
                        [(((Jz+M*L**2)+M*wheel_radius*L)/(wheel_radius*(M+2*m+(2*I/wheel_radius**2))*(Jz+M*L**2)-M*wheel_radius*L*M*L)), (((Jz+M*L**2)+M*wheel_radius*L)/(wheel_radius*(M+2*m+(2*I/wheel_radius**2))*(Jz+M*L**2)-M*wheel_radius*L*M*L))],
                        [0, 0],
                        [(-1*(M*L+wheel_radius*(M+2*m+(2*I/wheel_radius**2)))/(wheel_radius*(M+2*m+(2*I/wheel_radius**2))*(Jz+M*L**2)-M*wheel_radius*L*M*L)), (-1*(M*L+wheel_radius*(M+2*m+(2*I/wheel_radius**2)))/(wheel_radius*(M+2*m+(2*I/wheel_radius**2))*(Jz+M*L**2)-M*wheel_radius*L*M*L))],
                        [0, 0],
                        [(1/(wheel_radius*(m*D+(I*D)/wheel_radius**2+(2*Jy)/D))), -1/(wheel_radius*(m*D+(I*D)/wheel_radius**2+(2*Jy)/D))]
                        ])
        # lqr控制所需的Q、R矩阵
        Q = np.diag( [10,2,100,10,10,5] )
        R = np.diag( [1,1] )
        # 使用control库中的lqr函数来获取控制反馈增益K值
        K, S, E = control.lqr( A, B, Q, R ) #k为两行六列的矩阵
        K_L = K[0:1]  # 获取K的第一行六个元素作用于左轮
        K_R = K[1:2]  # 获取K的第二行六个元素作用于右轮
        return K_L,K_R

    def balance(self):
        if self.current_state is not None:
            L, M, self.pitch_com, roll_com,Jz,Jy=self.com(self.right_hip_position, self.right_knee_position, self.left_hip_position, self.left_knee_position, self.roll)
            # U=K*（Xd-X）
            K_L,K_R=self.K_cal(L,M,Jz,Jy)
            self.command_msg_L.data = np.matmul(K_L, (self.desired_state - self.current_state))
            self.command_msg_R.data = np.matmul(K_R, (self.desired_state - self.current_state))
            # 发布计算的平衡力矩值至力矩控制话题（即给机器人输入力矩值）
            left_command=-self.command_msg_L.data
            right_command=self.command_msg_R.data
            x_test = np.array([[self.slope_tra,self.current_position_x,self.velocity,self.ax,self.az,self.ay,self.pitch]])
            # x_test = np.array([[0.05,	-0.06,	0.34,	-0.93	,9.2	,0.015	,0.19]])
            # y_pred = self.data_NetworkFunction(x_test)
            # print("liju:",y_pred)
            # self.torque_lw_pred,self.torque_rw_pred=y_pred[0]
            self.command_pub_L3.publish(right_command)#话题订阅对象与仿真关节对象相反
            self.command_pub_R3.publish(left_command)#话题订阅对象与仿真关节对象相反
            # self.command_pub_L3.publish(self.torque_lw_pred)#话题订阅对象与仿真关节对象相反
            # self.command_pub_R3.publish(self.torque_rw_pred)#话题订阅对象与仿真关节对象相反


#vmc
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
        self.torque_lw = 0
        self.torque_rw = 0
        self.torque_lk = 0
        self.torque_rk = 0
        self.torque_lh = 0
        self.torque_rh = 0
        # 创建对象
        # 实例化双腿关节力矩控制话题的发布者对象
        self.command_pub_R1 = rospy.Publisher("/wheel_leg/joint1_position_controller/command", Float64, queue_size=10)
        self.command_pub_R2 = rospy.Publisher("/wheel_leg/joint2_position_controller/command", Float64, queue_size=10)
        self.command_pub_L1 = rospy.Publisher("/wheel_leg/joint4_position_controller/command", Float64, queue_size=10)
        self.command_pub_L2 = rospy.Publisher("/wheel_leg/joint5_position_controller/command", Float64, queue_size=10)
        
        # 实例化关节状态话题的订阅者对象
        self.state_sub    = rospy.Subscriber('/wheel_leg/joint_states', JointState, self.robot_state_callback)  # 获取机器人的位置与速度
        self.theta=rospy.Subscriber("/wheel_leg/imu", Imu, self.imu_data)
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
        
        self.torque_lw = data.effort[left_wheel_index]  # 左轮速度（弧度/秒）
        self.torque_rw = data.effort[right_wheel_index]  # 右轮速度（弧度/秒）
        self.torque_lk = data.effort[left_knee_index]  # 左膝速度（弧度/秒）
        self.torque_rk = data.effort[right_knee_index]  # 右膝速度（弧度/秒）
        self.torque_lh = data.effort[left_hip_index]  # 左髋速度（弧度/秒）
        self.torque_rh = data.effort[right_hip_index]  # 右髋速度（弧度/秒）
        # print(self.position_effort

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
        self.ax=a_x
        self.az=a_z



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
        self.command_pub_L1.publish(tao_l[0])
        self.command_pub_L2.publish(tao_l[1])
        self.command_pub_R1.publish(tao_r[0])
        self.command_pub_R2.publish(tao_r[1])
        
        # self.command_pub_L1.publish(0)
        # self.command_pub_L2.publish(0)
        # self.command_pub_R1.publish(0)
        # self.command_pub_R2.publish(0)


def main():
    a = VMC()
    b = LQR_Controller()
    first_time = True  # 控制是否输出列名

    while not rospy.is_shutdown():
        a.vmc()
        b.balance()

        if first_time:

           
            # print("sim_time, position_x, torque_lk, torque_lw, torque_rh, torque_rk, torque_rw, ax, az, slope_tra, velocity, roll_dot_dot, pitch_dot_dot, yaw_dot_dot, ay, position_lw, position_rw, current_position_x,yaw,yaw_dot,pitch,pitch_dot,roll,roll_dot,torque_lw_pred,torque_rw_pred")
            print("sim_time position_x current_position_x current_state "
                "torque_lh torque_lk torque_lw torque_rh torque_rk torque_rw "
                "ax ay az slope_tra velocity "
                "pitch yaw roll "
                "roll_dot pitch_dot yaw_dot "
                "roll_dot_dot pitch_dot_dot yaw_dot_dot "
                "position_lh position_lk position_lw position_rh position_rk position_rw "
                "velocity_lh velocity_lk velocity_lw velocity_rh velocity_rk velocity_rw "
                "wheel_position_x x y z vx vy vz")
            
            first_time = False

        # print(b.sim_time, b.position_x, a.torque_lk, a.torque_lw, a.torque_rh, a.torque_rk, a.torque_rw,
        #       b.ax, b.az, b.slope_tra, b.velocity, b.roll_dot_dot, b.pitch_dot_dot, b.yaw_dot_dot, b.ay,
        #       a.position_lw, a.position_rw, b.current_position_x,
        #       b.yaw,b.yaw_dot,b.pitch,b.pitch_dot,b.roll,b.roll_dot,b.torque_lw_pred,b.torque_rw_pred,)
        
        print(b.sim_time,b.position_x,b.current_position_x,b.current_state[0].item(),
              a.torque_lh,a.torque_lk,a.torque_lw,a.torque_rh,a.torque_rk,a.torque_rw,
              b.ax,b.ay,b.az,b.slope_tra,b.velocity,
              b.pitch,b.yaw,b.roll,
              b.roll_dot,b.pitch_dot,b.yaw_dot,
              b.roll_dot_dot,b.pitch_dot_dot,b.yaw_dot_dot,
              a.position_lh,a.position_lk,a.position_lw,a.position_rh,a.position_rk,a.position_rw,
              a.velocity_lh,a.velocity_lk,a.velocity_lw,a.velocity_rh,a.velocity_rk,a.velocity_rw,
              b.wheel_position_x,b.x,b.y,b.z,b.vx,b.vy,b.vz)
        # a.torque_L1,a.torque_L2,a.torque_R1,a.torque_R2,b.torque_L3,b.torque_R3
        b.rate.sleep()

if __name__ == '__main__':
    main()