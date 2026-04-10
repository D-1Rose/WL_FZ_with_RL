import copy
from controller_text import LQR_Controller
from controller_text import VMC
from num_envs_fz_control import BatchedLQRController
from num_envs_fz_control import BatchedVMC
import torch
import math
import genesis as gs # type: ignore
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat # type: ignore
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import numpy as np
import cv2

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower




class WheelLeggedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, 
                 domain_rand_cfg, terrain_cfg,
                   robot_morphs="urdf", show_viewer=True,
                     device="cuda:0", train_mode=True,control_mode=0,
                       pos_kp=100.0, pos_ki=5.0, pos_kd=20.0, vel_kp=50.0,
                         vel_ki=2.0, vel_kd=5.0,writer = None, step_count = None):
        self.device = torch.device(device)

        self.mode = train_mode   #True训练模式开启
        
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_slice_obs = obs_cfg["num_slice_obs"]
        self.history_length = obs_cfg["history_length"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  #输出维度，可自定义，这里暂时是2
        self.num_dofs = env_cfg["num_dofs"]  # 6
        self.num_commands = command_cfg["num_commands"]  # 4
        self.curriculum_cfg = curriculum_cfg
        self.domain_rand_cfg = domain_rand_cfg
        self.terrain_cfg = terrain_cfg
        self.num_respawn_points = self.terrain_cfg["num_respawn_points"]
        self.respawn_points = self.terrain_cfg["respawn_points"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.01  
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg  

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.noise = obs_cfg["noise"]
        
        # [新增] 初始化动作平滑 Buffer
        self.smooth_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.action_smooth_factor = 0.5  # 推荐 0.6~0.8。越大越平滑，反应越慢。
        self.common_step_counter = 0

        self.control_mode = control_mode  # 0=原控制，1=力矩控制
        # 初始化PID参数和状态
        self.pos_kp = pos_kp
        self.pos_ki = pos_ki
        self.pos_kd = pos_kd
        # 修正：初始化腿部关节PID状态（形状为 [num_envs, 4]）
        self.pos_integral = torch.zeros((self.num_envs, 4), device=self.device)  # 原 [4] → [1,4]（num_envs=1）
        self.last_pos_error = torch.zeros((self.num_envs, 4), device=self.device)

        self.vel_kp = vel_kp
        self.vel_ki = vel_ki
        self.vel_kd = vel_kd
        self.vel_integral = torch.zeros((self.num_envs, 2), device=self.device)  # 2个轮子
        self.last_vel_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.writer = writer  # 保存TensorBoard日志器
        self.step_count = step_count  # 保存步数计数器（注意：需要是可变对象，如列表）
        

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,  # Newton 迭代法
                enable_joint_limit=True,  # 关节限制
                batch_dofs_info=True,  # 配置批量处理自由度信息
                # batch_links_info=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="assets/terrain/plane/plane.urdf", fixed=True))
        # init roboot quat and pos
        match robot_morphs:
            case "urdf":
                self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"]["urdf"], device=self.device)
            case "mjcf":
                self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"]["mjcf"], device=self.device)
            case _:
                self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"]["urdf"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # add terrain 只能有一个Terrain(genesis v0.2.1)
        self.horizontal_scale = self.terrain_cfg["horizontal_scale"]  # 控制地形的水平尺寸（例如，每个像素对应实际世界中的 0.1 米）
        self.vertical_scale = self.terrain_cfg["vertical_scale"]
        # cv2.imread(..., cv2.IMREAD_GRAYSCALE) 返回一个二维数组，每个元素值范围为 0-255,白色区域（值 = 255）表示地形中的最高点，黑色区域（值 = 0）表示最低点
        self.height_field = cv2.imread("assets/terrain/png/"+self.terrain_cfg["train"]+".png", cv2.IMREAD_GRAYSCALE)  # 使用 OpenCV 读取 PNG 图像作为高度场数据
        self.terrain_height = torch.tensor(self.height_field, device=self.device) * self.vertical_scale
        if self.terrain_cfg["terrain"]:
            print("\033[1;35m open terrain\033[0m")
            if self.mode:
                self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                height_field = self.height_field,
                horizontal_scale=self.horizontal_scale, 
                vertical_scale=self.vertical_scale,
                ),)
                self.base_terrain_pos = torch.zeros((self.num_respawn_points, 3), device=self.device)  # 表示预定义的 复活点位置集合（x, y, z 坐标）。
                for i in range(self.num_respawn_points):
                    self.base_terrain_pos[i] = self.base_init_pos + torch.tensor(self.respawn_points[i], device=self.device)
                print("\033[1;34m respawn_points: \033[0m",self.base_terrain_pos)
            else:
                height_field = cv2.imread("assets/terrain/png/"+self.terrain_cfg["eval"]+".png", cv2.IMREAD_GRAYSCALE)
                self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                pos = (1.0,1.0,0.0),
                height_field = height_field,
                horizontal_scale=self.horizontal_scale, 
                vertical_scale=self.vertical_scale,
                ),)     
                print("\033[1;34m respawn_points: \033[0m",self.base_init_pos)

        # add robot
        base_init_pos = self.base_init_pos.cpu().numpy()
        if self.terrain_cfg["terrain"]:
            if self.mode:
                base_init_pos = self.base_terrain_pos[0].cpu().numpy()

        match robot_morphs:
            case "urdf":
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file="/home/huang/wheel_leg/wheel_legged_genesis_new/wheel_leg/src/description/urdf/wheel_leg.urdf",
                        pos = base_init_pos,
                        quat=self.base_init_quat.cpu().numpy(),
                    ),
                )
            case "mjcf":
                self.robot = self.scene.add_entity(
                    gs.morphs.MJCF(file="assets/mjcf/nz/nz_view.xml",
                    pos=base_init_pos),
                    vis_mode='collision'
                )
            case _:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file="/home/huang/wheel_leg/wheel_legged_genesis_new/wheel_leg/src/description/urdf/wheel_leg.urdf",
                        pos = base_init_pos,
                        quat=self.base_init_quat.cpu().numpy(),
                    ),
                )
        # build
        self.scene.build(n_envs=num_envs)

        # self.lqr_controller = LQR_Controller(robot=self.robot)
        # self.vmc_controller = VMC(robot=self.robot)

        self.lqr_controller = BatchedLQRController(num_envs=self.num_envs,
                                                   device=self.device)
        self.vmc_controller = BatchedVMC(num_envs=self.num_envs, 
                                         device=self.device)
        # self.fuzzy_gamma_d_g = torch.tensor(self.lqr_controller.fuzzy_gamma_d_g_default).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, 1)
        

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        # 创建统一的关节索引映射
        self.joint_name_to_idx = {
            "left_hip": self.motor_dofs[0],     # L1_joint
            "left_knee": self.motor_dofs[1],    # L2_joint
            "right_hip": self.motor_dofs[2],    # R1_joint
            "right_knee": self.motor_dofs[3],   # R2_joint
            "left_wheel": self.motor_dofs[4],   # L3_joint
            "right_wheel": self.motor_dofs[5],  # R3_joint
        }
        
        self.lqr_controller.set_joint_mapping(self.joint_name_to_idx)
        self.vmc_controller.set_joint_mapping(self.joint_name_to_idx)        

        # PD control parameters
        self.kp = np.full((self.num_envs, self.num_dofs), self.env_cfg["joint_kp"])
        self.kv = np.full((self.num_envs, self.num_dofs), self.env_cfg["joint_kv"])
        self.kp[:,4:6] = 0.0  # wheel kp is zero
        self.kv[:,4:6] = self.env_cfg["wheel_kv"]  # wheel kv is different from joint kv
        # self.robot.set_dofs_kp(self.kp, self.motor_dofs)
        # self.robot.set_dofs_kv(self.kv, self.motor_dofs)
        
        damping = np.full((self.num_envs, self.robot.n_dofs), self.env_cfg["damping"])
        damping[:,:6] = 0
        self.is_damping_descent = self.curriculum_cfg["damping_descent"]
        self.damping_max = self.curriculum_cfg["dof_damping_descent"][0]
        self.damping_min = self.curriculum_cfg["dof_damping_descent"][1]
        self.damping_step = self.curriculum_cfg["dof_damping_descent"][2]*(self.damping_max - self.damping_min)
        self.damping_threshold = self.curriculum_cfg["dof_damping_descent"][3]
        if self.is_damping_descent:
            self.damping_base = self.damping_max
        else:
            self.damping_base = self.env_cfg["damping"]
        self.robot.set_dofs_damping(damping, np.arange(0,self.robot.n_dofs))
        
        stiffness = np.full((self.num_envs,self.robot.n_dofs), self.env_cfg["stiffness"])
        stiffness[:,:6] = 0
        stiffness[:,self.motor_dofs[5]] = 0
        stiffness[:,self.motor_dofs[4]] = 0
        self.stiffness = self.domain_rand_cfg["dof_stiffness_descent"][0]
        self.stiffness_max = self.domain_rand_cfg["dof_stiffness_descent"][0]
        self.stiffness_end = self.domain_rand_cfg["dof_stiffness_descent"][1]
        self.robot.set_dofs_stiffness(stiffness, np.arange(0,self.robot.n_dofs))
        # from IPython import embed; embed()
        armature = np.full((self.num_envs, self.robot.n_dofs), self.env_cfg["armature"])
        armature[:,:6] = 0
        self.robot.set_dofs_armature(armature, np.arange(0, self.robot.n_dofs))
        

        #dof limits
        lower = [self.env_cfg["dof_limit"][name][0] for name in self.env_cfg["dof_names"]]
        upper = [self.env_cfg["dof_limit"][name][1] for name in self.env_cfg["dof_names"]]
        self.dof_pos_lower = torch.tensor(lower).to(self.device)
        self.dof_pos_upper = torch.tensor(upper).to(self.device)

        # set safe force
        lower = np.array([[-self.env_cfg["safe_force"][name] for name in self.env_cfg["dof_names"]] for _ in range(num_envs)])
        upper = np.array([[self.env_cfg["safe_force"][name] for name in self.env_cfg["dof_names"]] for _ in range(num_envs)])
        self.dof_force_lower = torch.tensor(lower).to(self.device)
        self.dof_force_upper = torch.tensor(upper).to(self.device)
        self.robot.set_dofs_force_range(
            lower          = torch.tensor(lower, device=self.device, dtype=torch.float32),
            upper          = torch.tensor(upper, device=self.device, dtype=torch.float32),
            dofs_idx_local = self.motor_dofs,
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)


        # prepare command_ranges lin_vel_x lin_vel_y ang_vel height_target
        # 创建一个三维张量，形状为 (环境数量, 命令数量, 2)，并初始化为0
        self.command_ranges = torch.zeros((self.num_envs, self.num_commands,2),device=self.device,dtype=gs.tc_float)
        self.command_ranges[:,0,0] = self.command_cfg["lin_vel_x_range"][0] * self.command_cfg["base_range"]
        self.command_ranges[:,0,1] = self.command_cfg["lin_vel_x_range"][1] * self.command_cfg["base_range"]
        self.command_ranges[:,1,0] = self.command_cfg["lin_vel_y_range"][0] * self.command_cfg["base_range"]
        self.command_ranges[:,1,1] = self.command_cfg["lin_vel_y_range"][1] * self.command_cfg["base_range"]
        self.command_ranges[:,2,0] = self.command_cfg["ang_vel_range"][0] * self.command_cfg["base_range"]
        self.command_ranges[:,2,1] = self.command_cfg["ang_vel_range"][1] * self.command_cfg["base_range"]
        self.height_range = self.command_cfg["height_target_range"][1]-self.command_cfg["height_target_range"][0]
        self.command_ranges[:,3,0] = self.command_cfg["height_target_range"][0] + self.height_range * (1 - self.command_cfg["base_range"])
        self.command_ranges[:,3,1] = self.command_cfg["height_target_range"][1]
        self.lin_vel_error = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)
        self.ang_vel_error = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)
        self.height_error = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)
        self.curriculum_lin_vel_scale = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)
        self.curriculum_ang_vel_scale = torch.zeros((self.num_envs,1), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

        self.slice_obs_buf = torch.zeros((self.num_envs, self.num_slice_obs), device=self.device, dtype=gs.tc_float)
        self.history_obs_buf = torch.zeros((self.num_envs, self.history_length, self.num_slice_obs), device=self.device, dtype=gs.tc_float)
        
        # [新增] 真实历史长缓存 (The Vault)
        # 我们需要存过去 30 帧，哪怕只用其中 5 帧，这样才能回溯到 0.3秒前
        self.true_history_len = 30
        self.true_history_buf = torch.zeros(
                    (self.num_envs, self.true_history_len, self.num_slice_obs), 
                    device=self.device, dtype=gs.tc_float
                )
        # [新增] 稀疏采样索引
        # index 1=t-1, 3=t-3, ..., 24=t-24
        self.history_indices = [1, 3, 6, 12, 24]
        # 校验配置
        if self.history_length != 5:#当前配置中history_length=5
             print(f"注意 : Config history_length={self.history_length}, 这表示选取几帧历史观测,在train配置中修改了history_length的话,需要增加选取哪几帧请在self.history_indices处修改选取的历史观测索引")

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.curriculum_rew_buf = torch.zeros_like(self.rew_buf)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["height_measurements"]], 
            device=self.device,
            dtype=gs.tc_float,
        )

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dofs = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros_like(self.dofs)
        self.dof_vel = torch.zeros_like(self.dofs)
        self.dof_force = torch.zeros_like(self.dofs)
        self.last_dof_vel = torch.zeros_like(self.dofs)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.basic_default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        
        default_dof_pos_list = [[self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]]] * self.num_envs
        self.default_dof_pos = torch.tensor(default_dof_pos_list,device=self.device,dtype=gs.tc_float,)
        init_dof_pos_list = [[self.env_cfg["joint_init_angles"][name] for name in self.env_cfg["dof_names"]]] * self.num_envs
        self.init_dof_pos = torch.tensor(init_dof_pos_list,device=self.device,dtype=gs.tc_float,)
        #膝关节
        # 创建左右膝的关节引用。
        self.left_knee = self.robot.get_joint("L2_joint")
        self.right_knee = self.robot.get_joint("R2_joint")
        # 创建张量存储膝关节的3D坐标
        self.left_knee_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.right_knee_pos = torch.zeros_like(self.left_knee_pos)
        # 创建所有机器人link与平面的接触力（3D）
        self.connect_force = torch.zeros((self.num_envs,self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)
        self.extras = dict()  # extra information for logging
        
        #跪地重启   注意是idx_local不需要减去base_idx
        # 与操作符号
        if(self.env_cfg["termination_if_base_connect_plane_than"]&self.mode):
            self.reset_links = [(self.robot.get_link(name).idx_local) for name in self.env_cfg["connect_plane_links"]]
            
        #域随机化 domain_rand_cfg
        # 设置随机化范围
        self.friction_ratio_low = self.domain_rand_cfg["friction_ratio_range"][0]
        self.friction_ratio_range = self.domain_rand_cfg["friction_ratio_range"][1] - self.friction_ratio_low
        self.base_mass_low = self.domain_rand_cfg["random_base_mass_shift_range"][0]
        self.base_mass_range = self.domain_rand_cfg["random_base_mass_shift_range"][1] - self.base_mass_low  
        self.other_mass_low = self.domain_rand_cfg["random_other_mass_shift_range"][0]
        self.other_mass_range = self.domain_rand_cfg["random_other_mass_shift_range"][1] - self.other_mass_low            
        self.dof_damping_low = self.domain_rand_cfg["damping_range"][0]
        self.dof_damping_range = self.domain_rand_cfg["damping_range"][1] - self.dof_damping_low
        self.dof_stiffness_low = self.domain_rand_cfg["dof_stiffness_range"][0]
        self.dof_stiffness_range = self.domain_rand_cfg["dof_stiffness_range"][1] - self.dof_stiffness_low
        if(self.dof_stiffness_low == 0) and (self.dof_stiffness_range == 0):
            self.is_stiffness = False
        else:
            self.is_stiffness = True      
        self.dof_armature_low = self.domain_rand_cfg["dof_armature_range"][0]
        self.dof_armature_range = self.domain_rand_cfg["dof_armature_range"][1] - self.dof_armature_low
        self.kp_low = self.domain_rand_cfg["random_KP"][0]
        self.kp_range = self.domain_rand_cfg["random_KP"][1] - self.kp_low
        self.kv_low = self.domain_rand_cfg["random_KV"][0]
        self.kv_range = self.domain_rand_cfg["random_KV"][1] - self.kv_low
        self.joint_angle_low = self.domain_rand_cfg["random_default_joint_angles"][0]
        self.joint_angle_range = self.domain_rand_cfg["random_default_joint_angles"][1] - self.joint_angle_low
        #地形训练索引
        self.terrain_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        # print("self.obs_buf.size(): ",self.obs_buf.size())
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        #外部力
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            rigid_solver = solver

        print("self.init_dof_pos",self.init_dof_pos)
        #初始化角度
        self.reset()
        
    def _resample_commands(self, envs_idx):
        # 为指定环境索引 envs_idx 重新采样控制指令（如线速度、角速度等）
        if self.command_cfg["limit_cmd_random"]:
            for idx in envs_idx:
                lin_x_low = self.command_ranges[idx, 0, 0]
                lin_x_high = self.command_ranges[idx, 0, 1]
                #以高斯分布随机给上下限范围内的控制指令
                self.commands[idx, 0] = gs_rand_float(lin_x_low, lin_x_high, (1,), self.device)  

                # [核心修改]：人为注入“冲锋偏见”
                # 生成一个随机概率 [0, 1]
                rand_prob = torch.rand(1, device=self.device)
                
                # --- 模式 A: 40% 概率向前冲 (负压) ---
                if rand_prob < 0.4:
                    self.commands[idx, 0] = gs_rand_float(-2.0, -0.5, (1,), self.device)
                    
                # --- 模式 B: 40% 概率向后冲 (正压) ---
                # [新增] 专门训练后退爬坡，或者在坡上刹住不滑下来
                elif rand_prob < 0.8:
                     self.commands[idx, 0] = gs_rand_float(0.5, 2.0, (1,), self.device)
                     
                # --- 模式 C: 20% 概率低速/停车 (精细操作) ---
                else:
                    # 采样接近 0 的速度，训练静态平衡
                    self.commands[idx, 0] = gs_rand_float(-0.5, 0.5, (1,), self.device)

                # 40% 的概率：保持配置文件里的全范围 ([-1, 1])，让它复习怎么停车和后退
                # (不做任何覆盖，保留上面的随机采样结果)
                # # 计算角速度上限（基于线速度动态调整）

                ang_vel_max = (self.command_cfg["wheel_max_w"]*2*self.command_cfg["wheel_radius"] - torch.abs(self.commands[idx, 0]) * 2)/self.command_cfg["wheel_spacing"]
                ang_vel_low = self.command_ranges[idx, 2, 0]
                ang_vel_high = self.command_ranges[idx, 2, 1]
                ang_vel_low.clamp_min_(-ang_vel_max)
                ang_vel_high.clamp_max_(ang_vel_max)
                self.commands[idx, 2] = gs_rand_float(ang_vel_low, ang_vel_high, (1,), self.device)
                self.commands[idx, 1] = gs_rand_float(self.command_ranges[idx, 1, 0], self.command_ranges[idx, 1, 1], (1,), self.device)
                self.commands[idx, 3] = gs_rand_float(self.command_ranges[idx, 3, 0], self.command_ranges[idx, 3, 1], (1,), self.device)
        else:
            for idx in envs_idx:
                for command_idx in range(self.num_commands):
                    low = self.command_ranges[idx, command_idx, 0]
                    high = self.command_ranges[idx, command_idx, 1]
                    self.commands[idx, command_idx] = gs_rand_float(low, high, (1,), self.device)
        self.commands[envs_idx, 1]=0

    def set_commands(self,envs_idx,commands):
        self.commands[envs_idx]=torch.tensor(commands,device=self.device, dtype=gs.tc_float)




    def step(self, actions):

        # 限制范围 (Raw Input)
        raw_actions = torch.tanh(actions)
        
        # 低通滤波 (Smoothing)
        # self.actions 永远存储"当前这一帧平滑后的理想动作"
        # 公式：y_t = 0.6 * y_{t-1} + 0.4 * x_t
        self.actions = self.action_smooth_factor * self.actions + \
                       (1 - self.action_smooth_factor) * raw_actions
        
        # 延迟模拟 (Latency Simulation)
        # 逻辑：如果是延迟模式，机器人执行的是"上一帧存下来的动作"
        if self.simulate_action_latency:
            real_actions = self.last_actions.clone()      # 取出存货去执行 (Real Execution)
            self.last_actions = self.actions.clone()      # 把当前货存入仓库 (Buffer Update)
        else:
            real_actions = self.actions.clone()           # 没延迟，直接执行


        action_gama = real_actions[:, 0]
        action_eth = real_actions[:, 1]
        action_ld_max = real_actions[:, 2]
        action_ld_min = real_actions[:, 3]
        action_a = real_actions[:, 4]
        action_pitch_kp = real_actions[:, 5]
        action_pitch_kd = real_actions[:, 6]
        action_vel_kp = real_actions[:, 7]
        action_vel_ki = real_actions[:, 8]
        # action_pitch_target = real_actions[:, 9]
        # ==================================================================================
        #  参数解算 (使用 real_actions)
        # ==================================================================================
        
        # --- Part A: 模糊参数 (Residual Offset: Default + Action * Delta) ---
        # 对应动作维度 [0, 1, 2, 3, 4]
        new_gamma  = self.lqr_controller.fuzzy_gamma_d_g_default + action_gama * self.lqr_controller.gamma_d_g_delta
        new_eth    = self.lqr_controller.fuzzy_eth_default       + action_eth * self.lqr_controller.eth_delta
        new_ld_max = self.lqr_controller.fuzzy_ld_max_default    + action_ld_max * self.lqr_controller.ld_max_delta
        new_ld_min = self.lqr_controller.fuzzy_ld_min_default    + action_ld_min * self.lqr_controller.ld_min_delta
        new_a      = self.lqr_controller.fuzzy_a_default         + action_a * self.lqr_controller.a_delta

        # --- Part B: 增益参数 (Exponential Mapping: Default * Scale ^ Action) ---
        # [核心修改] 定义不同的缩放因子
        # 1. Kp (主力): 给大范围探索 (0.2x ~ 5.0x)
        scale_pitch_kp = 20
        scale_vel_kp   = 20
        
        # 2. Kd (敏感): 范围收窄，防止微分噪声爆炸 (0.33x ~ 3.0x)
        scale_pitch_kd = 4
        
        # 3. Ki (成长): 需要从微小种子长大，给大倍数 (0.1x ~ 10.0x)
        # 假设 default=0.1，action=+1 -> Ki=1.0; action=-1 -> Ki=0.01
        scale_vel_ki   = 5

        # --- 计算参数 ---
        # Pitch Loop
        new_pitch_kp = self.lqr_controller.pitch_kp_default * torch.pow(scale_pitch_kp, real_actions[:, 5])
        new_pitch_kd = self.lqr_controller.pitch_kd_default * torch.pow(scale_pitch_kd, real_actions[:, 6]) # [新增]
        
        # Velocity Loop
        new_vel_kp   = self.lqr_controller.vel_kp_default   * torch.pow(scale_vel_kp,   real_actions[:, 7])
        new_vel_ki   = self.lqr_controller.vel_ki_default   * torch.pow(scale_vel_ki,   real_actions[:, 8]) # [新增]
        
        # # --- 安全截断 (根据实际物理经验微调) ---
        # new_pitch_kp = torch.clamp(new_pitch_kp, 0.01, 10)
        # new_vel_kp   = torch.clamp(new_vel_kp,   0.1,  100)
        
        # # Kd 绝不能太大，否则仿真会炸飞
        # new_pitch_kd = torch.clamp(new_pitch_kd, 0.0,  2.0)  
        
        # # Ki 需要防积分饱和，上限适中即可
        # new_vel_ki   = torch.clamp(new_vel_ki,   0.0,  50)



        # new_pitch_target = self.lqr_controller.pitch_d_default + real_actions[:, 9] * self.lqr_controller.pitch_d_delta


        # 更新模糊参数（自动限制范围）
        self.lqr_controller.set_fuzzy_params(new_gamma, 
                                             new_eth, 
                                             new_ld_max, 
                                             new_ld_min, 
                                             new_a , 
                                             new_pitch_kp, 
                                             new_pitch_kd , 
                                             new_vel_kp ,
                                             new_vel_ki , 
                                            #  new_pitch_target
                                             )
        
        self.lqr_controller.set_commands(
            velocity_d = self.commands[:, 0],
            yaw_d_dot = self.commands[:, 2],

        )


        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        # =============  统一获取所有状态（只调用一次！） =============
        all_dof_pos = self.robot.get_dofs_position()  # [num_envs, n_dofs]
        all_dof_vel = self.robot.get_dofs_velocity()  # [num_envs, n_dofs]
        base_vel = self.base_lin_vel              # [num_envs, 3]
        base_quat = self.robot.get_quat()             # [num_envs, 4]
        base_euler = self.base_euler               # [num_envs, 3]
        base_ang = self.base_ang_vel               # [num_envs, 3]
        inv_base_quat = inv_quat(self.base_quat)
        
        # =============  传递状态给控制器 =============
        self.lqr_controller.set_state(all_dof_pos, all_dof_vel, base_vel, base_euler, base_ang)
        self.vmc_controller.set_state(all_dof_pos, all_dof_vel, base_vel, base_euler, base_ang)
        leg_torques = self.vmc_controller.vmc()      # [num_envs, 4]
        wheel_torques = self.lqr_controller.balance()  # [num_envs, 2]        
        # print(f"leg_torques: {leg_torques}")
        # print(f"wheel_torques: {wheel_torques}")

        # ============= 传递力矩给机器人 =============
        self.robot.control_dofs_force(leg_torques, self.motor_dofs[0:4])
        self.robot.control_dofs_force(wheel_torques, self.motor_dofs[4:6])

        self.scene.step()


        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.get_relative_terrain_pos(self.robot.get_pos())
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_lin_acc[:] = (self.base_lin_vel[:] - self.last_base_lin_vel[:])/ self.dt
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat) 
        self.base_ang_acc[:] = (self.base_ang_vel[:] - self.last_base_ang_vel[:]) / self.dt
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat) 
        # =======================================================================
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs) 
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs) 
        self.dof_force[:] = self.robot.get_dofs_force(self.motor_dofs)
        if self.noise["use"]:
            self.base_ang_vel[:] += torch.randn_like(self.base_ang_vel) * self.noise["ang_vel"][0] + (torch.rand_like(self.base_ang_vel)*2-1) * self.noise["ang_vel"][1]
            self.projected_gravity += torch.randn_like(self.projected_gravity) * self.noise["gravity"][0] + (torch.rand_like(self.projected_gravity)*2-1) * self.noise["gravity"][1]
            self.dof_pos[:] += torch.randn_like(self.dof_pos) * self.noise["dof_pos"][0] + (torch.rand_like(self.dof_pos)*2-1) * self.noise["dof_pos"][1]
            self.dof_vel[:] += torch.randn_like(self.dof_vel) * self.noise["dof_vel"][0] + (torch.rand_like(self.dof_vel)*2-1) * self.noise["dof_vel"][1]
        
        #获取膝关节高度
        self.left_knee_pos[:] = self.left_knee.get_pos()
        self.right_knee_pos[:] = self.right_knee.get_pos()
        #碰撞力
        self.connect_force = self.robot.get_links_net_contact_force()

        # update last
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:]
        
        #步数
        self.episode_lengths += 1

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        # check terrain_buf
        # 线速度达到预设的90%范围，角速度达到90%以上去其他地形(建议高一点)
        self.terrain_buf = self.command_ranges[:, 0, 1] > self.command_cfg["lin_vel_x_range"][1] * 0.9
        self.terrain_buf &= self.command_ranges[:, 2, 1] > self.command_cfg["ang_vel_range"][1] * 0.9
        #固定一部分去地形
        self.terrain_buf[:int(self.num_envs*0.4)] = 1
        
        # check termination and reset
        if(self.mode):
            self.check_termination()

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        if(self.mode):
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        if(self.mode):
            self.rew_buf[:] = 0.0
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew
            
        # compute curriculum reward
        self.lin_vel_error += torch.abs(self.commands[:, :2] - self.base_lin_vel[:, :2]).mean(dim=1, keepdim=True)
        self.ang_vel_error += torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2]).mean()
        self.height_error += torch.abs(self.commands[:, 3] - self.base_pos[:, 2]).mean()

        if(self.mode):
            self._resample_commands(envs_idx)
            # self.curriculum_commands()
        # else:
        #     print("base_lin_vel: ",self.base_lin_vel[0,:])
        



    # ============= 观测计算（含模糊参数） =============
        fuzzy_params = self.lqr_controller.current_fuzzy_params  # [gamma_d_g, eth, ld_max, ld_min, a]
        # 1. 确保 pitch 和 pitch_dot 都是 (N, 1) 的形状，然后拼接成 (N, 2)
        pitch_obs = torch.cat(
            [
                self.lqr_controller.pitch.unsqueeze(-1),
                self.lqr_controller.pitch_dot.unsqueeze(-1),
            ],
            axis=-1
        ) # 形状现在是 (N, 2)
    
        # compute observations
        self.slice_obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 4
                (self.dof_pos[:,0:4] - self.default_dof_pos[:,0:4]) * self.obs_scales["dof_pos"],  # 4
                self.dof_vel * self.obs_scales["dof_vel"],  # 6
                self.actions,  #9
                fuzzy_params,  # 9
                pitch_obs, # 2
            ],
            axis=-1,
        )
        # self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # print("slice_obs_buf: ",self.slice_obs_buf)
        
        # # Combine the current observation with historical observations (e.g., along the time axis)
        # self.obs_buf = torch.cat([self.history_obs_buf, self.slice_obs_buf.unsqueeze(1)], dim=1).view(self.num_envs, -1)
        # # Update history buffer（尾部插入）
        # if self.history_length > 1:
        #     self.history_obs_buf[:, :-1, :] = self.history_obs_buf[:, 1:, :].clone() # 移位操作
        # self.history_obs_buf[:, -1, :] = self.slice_obs_buf 

        # ================== [开始修改] Strided History 逻辑 ==================
        
        # 1. 更新真实长缓存 (FIFO 队列整体向后移位)
        # 腾出 index 0 的位置给最新帧（头部插入）
        self.true_history_buf[:, 1:, :] = self.true_history_buf[:, :-1, :].clone()
        
        # 2. 将当前最新帧存入 index 0
        self.true_history_buf[:, 0, :] = self.slice_obs_buf 
        
        # 3. 稀疏采样 (只取我们关注的那 5 帧: t-1, t-3, t-6, t-12, t-24)
        # shape: [num_envs, 5, num_slice_obs]
        strided_history = self.true_history_buf[:, self.history_indices, :]
        
        # 4. 展平历史数据
        # shape: [num_envs, 5 * num_slice_obs]
        flat_history = strided_history.reshape(self.num_envs, -1)
        
        # 5. 拼接最终 Observation
        # 结构: [历史5帧 (t-24...t-1), 当前帧 (t)]
        # 这样既保留了长时记忆，又没有维度冗余
        self.obs_buf = torch.cat([flat_history, self.slice_obs_buf], dim=-1)

        # ================== [修改结束] ==================


        # =================================================================================
        # [新增] TensorBoard 数据记录模块 (仅在非训练模式且 writer 存在时触发)
        # =================================================================================
        if not self.mode and self.writer is not None and self.step_count is not None:
            # 获取当前的全局步数
            current_step = self.step_count[0]
            
            # --- 1. 核心跟踪性能 ---
            # 记录 X 方向速度误差 (Command - Actual)
            vel_error = self.commands[0, 0] - self.base_lin_vel[0, 0]
            self.writer.add_scalar('Performance/Vel_Error_X', vel_error.item(), current_step)
            self.writer.add_scalar('Performance/Vel_Actual_X', self.base_lin_vel[0, 0].item(), current_step)
            self.writer.add_scalar('Performance/Vel_Command_X', self.commands[0, 0].item(), current_step)
            
            # 记录 Pitch 姿态 (看是否震荡)
            # 注意：lqr_controller.pitch 是弧度，建议转角度显示，直观
            pitch_deg = self.lqr_controller.pitch[0].item() * 1
            self.writer.add_scalar('Performance/Pitch_Deg', pitch_deg, current_step)
            
            # --- 2. 控制器内部状态 (调试的关键) ---
            # 记录实时变化的 PID 参数 (验证增量式调优是否生效)
            # 假设第0个环境
            self.writer.add_scalar('Controller/Pitch_Kp', self.lqr_controller.pitch_kp[0].item(), current_step)
            self.writer.add_scalar('Controller/Pitch_Kd', self.lqr_controller.pitch_kd[0].item(), current_step)
            self.writer.add_scalar('Controller/Vel_Kp', self.lqr_controller.vel_kp[0].item(), current_step)
            self.writer.add_scalar('Controller/Vel_Ki', self.lqr_controller.vel_ki[0].item(), current_step)
            
            # 记录模糊参数 (验证自适应逻辑)
            # fuzzy_params 顺序: [gamma, eth, ld_max, ld_min, a]
            fuzzy = self.lqr_controller.current_fuzzy_params[0]
            self.writer.add_scalar('Fuzzy/Gamma', fuzzy[0].item(), current_step)
            self.writer.add_scalar('Fuzzy/Eth', fuzzy[1].item(), current_step)
            
            # --- 3. 机械健康度 & 平滑度 ---
            # 动作平滑度 (L2 范数)
            action_diff = torch.norm(self.actions[0] - self.last_actions[0])*100
            self.writer.add_scalar('Stability/Action_Smoothness', action_diff.item(), current_step)
            
            # 关节加速度 (震动指标)
            dof_acc = torch.mean(torch.abs((self.dof_vel[0] - self.last_dof_vel[0]) / self.dt))*100
            self.writer.add_scalar('Stability/Joint_Acc_Mean', dof_acc.item(), current_step)

            # 步数自增 (非常重要！否则所有点都在同一个 x 轴位置)
            self.step_count[0] += 1





        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras



    

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # print("\033[31m Reset Reset Reset Reset Reset Reset\033[0m")
        # reset dofs
        self.dof_pos[envs_idx] = self.init_dof_pos[envs_idx]  # 根据环境的索引然后利用init_dof_pos进行重置（有默认值）
        self.dof_vel[envs_idx] = 0.0  # 速度全部是0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)  # 将指定的环境的初始姿态进行重置
        if self.terrain_cfg["terrain"]:  # 如果使用了地形
            if self.mode:
                terrain_buf = self.terrain_buf[envs_idx]  # 标记每个环境是否使用地形
                # 获取非零元素的索引并展平为一维张量
                terrain_idx = envs_idx[terrain_buf.nonzero(as_tuple=False).flatten()] # 获取 envs_idx 中满足条件的索引
                non_terrain_idx = envs_idx[(~terrain_buf).nonzero(as_tuple=False).flatten()] # 获取 envs_idx 中不满足条件的索引
                # 设置地形位置
                if len(terrain_idx) > 0: # 只有当有满足地形重置条件的环境时才执行
                    #目前认为坡路和崎岖路面是相同难度，所以reset随机选取一个环境去复活
                    n = len(terrain_idx)
                    random_idx = torch.randint(1, self.num_respawn_points, (n,)) # 注意从 1 开始，避免使用 base_terrain_pos[0] 作为随机位置
                    # 选择随机位置,
                    selected_pos = self.base_terrain_pos[random_idx]
                    self.base_pos[terrain_idx] = selected_pos
                # 设置非地形位置 (默认位置)
                if len(non_terrain_idx) > 0:
                    self.base_pos[non_terrain_idx] = self.base_terrain_pos[0]
            else:
                self.base_pos[envs_idx] = self.base_init_pos
        else:
            self.base_pos[envs_idx] = self.base_init_pos   #没开地形就基础
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # ============= 新增：重置控制器状态 =============
        # 重置 LQR 控制器的 PID 积分项
        self.lqr_controller.command_pitch.prev_error[envs_idx] = 0.0
        self.lqr_controller.command_pitch.integral[envs_idx] = 0.0
        
        self.lqr_controller.command_velocity.prev_error[envs_idx] = 0.0
        self.lqr_controller.command_velocity.integral[envs_idx] = 0.0
        
        self.lqr_controller.command_yaw_dot.prev_error[envs_idx] = 0.0
        self.lqr_controller.command_yaw_dot.integral[envs_idx] = 0.0

        # [新增/修复] 必须清空历史记忆！防止机器人带着上一局跌倒的残影开始新的一局
        self.true_history_buf[envs_idx] = 0.0

        
        # 重置模糊参数为默认值
        self.lqr_controller.fuzzy_gamma_d_g[envs_idx] = self.lqr_controller.fuzzy_gamma_d_g_default[envs_idx]
        self.lqr_controller.fuzzy_eth[envs_idx] = self.lqr_controller.fuzzy_eth_default[envs_idx]
        self.lqr_controller.fuzzy_ld_max[envs_idx] = self.lqr_controller.fuzzy_ld_max_default[envs_idx]
        self.lqr_controller.fuzzy_ld_min[envs_idx] = self.lqr_controller.fuzzy_ld_min_default[envs_idx]
        self.lqr_controller.fuzzy_a[envs_idx] = self.lqr_controller.fuzzy_a_default[envs_idx]
        
        self.lqr_controller.pitch_kp[envs_idx] = self.lqr_controller.pitch_kp_default[envs_idx]
        self.lqr_controller.vel_kp[envs_idx] = self.lqr_controller.vel_kp_default[envs_idx]
        self.lqr_controller.pitch_kd[envs_idx] = self.lqr_controller.pitch_kd_default[envs_idx]
        self.lqr_controller.vel_ki[envs_idx] = self.lqr_controller.vel_ki_default[envs_idx]
        # self.lqr_controller.pitch_d[envs_idx] = self.lqr_controller.pitch_d_default[envs_idx]

        # 更新 current_fuzzy_params
        self.lqr_controller.current_fuzzy_params = torch.stack([
            self.lqr_controller.fuzzy_gamma_d_g,
            self.lqr_controller.fuzzy_eth,
            self.lqr_controller.fuzzy_ld_max,
            self.lqr_controller.fuzzy_ld_min,
            self.lqr_controller.fuzzy_a,
            self.lqr_controller.pitch_kp,
            self.lqr_controller.pitch_kd,
            self.lqr_controller.vel_kp,
            self.lqr_controller.vel_ki,
            # self.lqr_controller.pitch_d
        ], dim=1)


        # reset buffers
        # [修改] 重置动作 Buffer
        self.actions[envs_idx] = 0.0      # 重置平滑器
        self.last_actions[envs_idx] = 0.0 # 重置延迟队列
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True



        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)  
        if self.mode:
            self.domain_rand(envs_idx)
        self.episode_lengths[envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True  # 用于标志环境是否需要reset（全部）
        self.reset_idx(torch.arange(self.num_envs, device=self.device))  # 根据索引去reset环境
        return self.obs_buf, None

    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # self.reset_buf |= torch.abs(self.base_pos[:, 2]) < self.env_cfg["termination_if_base_height_greater_than"]
        #特殊姿态重置
        # self.reset_buf |= torch.abs(self.left_knee_pos[:,2]) < self.env_cfg["termination_if_knee_height_greater_than"]
        # self.reset_buf |= torch.abs(self.right_knee_pos[:,2]) < self.env_cfg["termination_if_knee_height_greater_than"]
        if(self.env_cfg["termination_if_base_connect_plane_than"]):
            for idx in self.reset_links:
                self.reset_buf |= torch.abs(self.connect_force[:,idx,:]).sum(dim=1) > 0
        
    def domain_rand(self, envs_idx):
        # 随机化摩擦系数,self.robot.n_links=7(base_link+6links)
        friction_ratio = self.friction_ratio_low + self.friction_ratio_range * torch.rand(len(envs_idx), self.robot.n_links)
        self.robot.set_friction_ratio(friction_ratio=friction_ratio,
                                      link_indices=np.arange(0, self.robot.n_links),
                                      envs_idx = envs_idx)

        base_mass_shift = self.base_mass_low + self.base_mass_range * torch.rand(len(envs_idx), 1, device=self.device)
        other_mass_shift =-self.other_mass_low + self.other_mass_range * torch.rand(len(envs_idx), self.robot.n_links - 1, device=self.device)
        mass_shift = torch.cat((base_mass_shift, other_mass_shift), dim=1)
        self.robot.set_mass_shift(mass_shift=mass_shift,
                                  link_indices=np.arange(0, self.robot.n_links),
                                  envs_idx = envs_idx)

        base_com_shift = -self.domain_rand_cfg["random_base_com_shift"] / 2 + self.domain_rand_cfg["random_base_com_shift"] * torch.rand(len(envs_idx), 1, 3, device=self.device)
        other_com_shift = -self.domain_rand_cfg["random_other_com_shift"] / 2 + self.domain_rand_cfg["random_other_com_shift"] * torch.rand(len(envs_idx), self.robot.n_links - 1, 3, device=self.device)
        com_shift = torch.cat((base_com_shift, other_com_shift), dim=1)
        self.robot.set_COM_shift(com_shift=com_shift,
                                 link_indices=np.arange(0, self.robot.n_links),
                                 envs_idx = envs_idx)

        # print(f"kp_low: {self.kp_low}, kp_range:{self.kp_range}, envs_idx:{envs_idx}")
        kp_shift = (self.kp_low + self.kp_range * torch.rand(len(envs_idx), self.num_dofs)) * self.kp[0]
        # print(f"kpshift:{kp_shift}")
        self.robot.set_dofs_kp(kp_shift, self.motor_dofs, envs_idx=envs_idx)
        
        kv_shift = (self.kv_low + self.kv_range * torch.rand(len(envs_idx), self.num_dofs)) * self.kv[0]
        self.robot.set_dofs_kv(kv_shift, self.motor_dofs, envs_idx = envs_idx)
        
        #random_default_joint_angles
        dof_pos_shift = self.joint_angle_low + self.joint_angle_range * torch.rand(len(envs_idx),self.num_dofs,device=self.device,dtype=gs.tc_float)
        self.default_dof_pos[envs_idx] = dof_pos_shift + self.basic_default_dof_pos
        
        # damping下降
        if self.is_damping_descent:
            if self.episode_lengths[envs_idx].mean()/(self.env_cfg["episode_length_s"]*self.stiffness_end/self.dt) > self.damping_threshold:
                self.damping_base -= self.damping_step
                if self.damping_base < self.damping_min:
                    self.damping_base = self.damping_min
            else:
                self.damping_base += self.damping_step
                if self.damping_base > self.damping_max:
                    self.damping_base = self.damping_max
        damping = (self.dof_damping_low+self.dof_damping_range * torch.rand(len(envs_idx), self.robot.n_dofs)) * self.damping_base
        damping[:,:6] = 0  # damping:[env_idx, 12]
        self.robot.set_dofs_damping(damping=damping,
                                   dofs_idx_local=np.arange(0, self.robot.n_dofs),
                                   envs_idx=envs_idx)
        
        if(self.is_stiffness):
            stiffness = (self.dof_stiffness_low+self.dof_stiffness_range * torch.rand(len(envs_idx), self.robot.n_dofs))
            stiffness[:,self.robot.n_dofs-6:] = 0
            stiffness[:,self.motor_dofs[5]] = 0
            stiffness[:,self.motor_dofs[4]] = 0
            self.robot.set_dofs_stiffness(stiffness=stiffness,
                                       dofs_idx_local=np.arange(0, self.robot.n_dofs),
                                       envs_idx=envs_idx)
        else:
            #刚度下降
            stiffness_ratio = 1 - (self.episode_lengths[envs_idx].mean()/(self.env_cfg["episode_length_s"]
                                                                          *self.stiffness_end/self.dt))
            if stiffness_ratio < 0:
                stiffness_ratio = 0
            self.stiffness = stiffness_ratio*self.stiffness_max
            stiffness = torch.full((len(envs_idx), self.robot.n_dofs),self.stiffness)
            stiffness[:,self.robot.n_dofs-6:] = 0
            stiffness[:,self.motor_dofs[5]] = 0
            stiffness[:,self.motor_dofs[4]] = 0
            self.robot.set_dofs_stiffness(stiffness=stiffness,
                                       dofs_idx_local=np.arange(0, self.robot.n_dofs),
                                       envs_idx=envs_idx)
        
        armature = (self.dof_armature_low+self.dof_armature_range * torch.rand(len(envs_idx), self.robot.n_dofs))
        armature[:,:6] = 0
        self.robot.set_dofs_armature(armature=armature,
                                   dofs_idx_local=np.arange(0, self.robot.n_dofs),
                                   envs_idx=envs_idx)

    def adjust_scale(self, error, lower_err, upper_err, scale, scale_step, min_range, range_cfg):
        # 计算误差范围
        min_condition = error < lower_err
        max_condition = error > upper_err
        # 调整 scale
        scale[min_condition] += scale_step
        scale[max_condition] -= scale_step
        scale.clip_(min_range, 1)
        # 更新 command_ranges
        range_min, range_max = range_cfg
        return scale * range_min, scale * range_max

    def curriculum_commands(self, num):
        # 更新误差
        self.lin_vel_error /= num
        self.ang_vel_error /= num
        self.height_error /= num
        # 调整线速度
        lin_min_range, lin_max_range = self.adjust_scale(
            self.lin_vel_error, 
            self.curriculum_cfg["lin_vel_err_range"][0],   #误差反馈更新
            self.curriculum_cfg["lin_vel_err_range"][1],    #err back update
            self.curriculum_lin_vel_scale, 
            self.curriculum_cfg["curriculum_lin_vel_step"], 
            self.curriculum_cfg["curriculum_lin_vel_min_range"], 
            self.command_cfg["lin_vel_x_range"]
        )
        self.command_ranges[:, 0, 0] = lin_min_range.squeeze()
        self.command_ranges[:, 0, 1] = lin_max_range.squeeze()
        # 调整角速度    角速度误差可以大一些，因为command范围更大
        ang_min_range, ang_max_range = self.adjust_scale(
            self.ang_vel_error, 
            self.curriculum_cfg["ang_vel_err_range"][0],
            self.curriculum_cfg["ang_vel_err_range"][1],
            self.curriculum_ang_vel_scale, 
            self.curriculum_cfg["curriculum_ang_vel_step"], 
            self.curriculum_cfg["curriculum_ang_vel_min_range"], 
            self.command_cfg["ang_vel_range"]
        )
        self.command_ranges[:, 2, 0] = ang_min_range.squeeze()
        self.command_ranges[:, 2, 1] = ang_max_range.squeeze()
        #调整高度
        add_height = self.height_error.squeeze() > 0.1
        self.command_ranges[add_height,3,0] += self.curriculum_cfg["curriculum_height_target_step"]
        cut_height = self.height_error.squeeze() < 0.05
        self.command_ranges[cut_height,3,0] -= self.curriculum_cfg["curriculum_height_target_step"]
        self.command_ranges[:,3,0].clip_(self.command_cfg["height_target_range"][0],
                                         self.command_cfg["height_target_range"][0] + self.height_range * (1-self.command_cfg["base_range"]))
        # 重置误差
        self.lin_vel_error = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.ang_vel_error = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.height_error = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)

    def get_relative_terrain_pos(self, base_pos):  # 双线性插值算法，用于计算机器人在不规则地形上的相对高度
        if not self.terrain_cfg["terrain"]:
            return base_pos
        #对多个 (x, y) 坐标进行双线性插值计算地形高度
        # 提取x和y坐标
        x = base_pos[:, 0]
        y = base_pos[:, 1]
        # 转换为浮点数索引
        fx = x / self.horizontal_scale
        fy = y / self.horizontal_scale
        # 获取四个最近的整数网格点，确保在有效范围内
        x0 = torch.floor(fx).int()
        x1 = torch.min(x0 + 1, torch.full_like(x0, self.terrain_height.shape[1] - 1))
        y0 = torch.floor(fy).int()
        y1 = torch.min(y0 + 1, torch.full_like(y0, self.terrain_height.shape[0] - 1))
        # 确保x0, x1, y0, y1在有效范围内
        x0 = torch.clamp(x0, 0, self.terrain_height.shape[1] - 1)
        x1 = torch.clamp(x1, 0, self.terrain_height.shape[1] - 1)
        y0 = torch.clamp(y0, 0, self.terrain_height.shape[0] - 1)
        y1 = torch.clamp(y1, 0, self.terrain_height.shape[0] - 1)
        # 获取四个点的高度值
        # 使用广播机制处理批量数据
        Q11 = self.terrain_height[y0, x0]
        Q21 = self.terrain_height[y0, x1]
        Q12 = self.terrain_height[y1, x0]
        Q22 = self.terrain_height[y1, x1]
        # 计算双线性插值
        wx = fx - x0
        wy = fy - y0
        height = (
            (1 - wx) * (1 - wy) * Q11 +
            wx * (1 - wy) * Q21 +
            (1 - wx) * wy * Q12 +
            wx * wy * Q22
        )
        base_pos[:,2] -= height
        return base_pos


# # ------------ reward functions----------------
# ========================================================================
    # 1. 性能维度 (Performance): 驱动力
    # ========================================================================
    def _reward_tracking_lin_vel(self):
        # 线速度跟踪：这是唯一的“正分”来源，必须强势
        # Sigma = 0.15 意味着误差 > 0.15m/s 时奖励迅速衰减，逼迫 RL 精细调参
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / 0.15)

    def _reward_tracking_ang_vel(self):
        # 角速度跟踪
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    # ========================================================================
    # 2. 机械健康与姿态 (Mechanical Health & Pose): 你强调的部分
    # ========================================================================
    def _reward_joint_pose_health(self):
        # [新增核心] 关节自然姿态引导
        # 惩罚关节偏离 default_dof_pos 的程度。
        # 作用：防止机器人为了加速而出现“劈岔”、“后坐”或“奇怪扭曲”的姿态。
        # 只取前4个关节(髋/膝)，轮子(4,5)不需要此约束
        joint_error = torch.sum(torch.square(self.dof_pos[:, :4] - self.default_dof_pos[:, :4]), dim=1)
        return torch.exp(-joint_error / 0.5)

    def _reward_knee_height(self):
        # [保留] 防止跪地
        # 膝盖高度低于 8cm 视为极度危险
        left_knee_low = self.left_knee_pos[:, 2] < 0.08
        right_knee_low = self.right_knee_pos[:, 2] < 0.08
        return (left_knee_low.float() + right_knee_low.float())

    def _reward_lqr_pitch_position(self):
        # [适配 9维动作]
        # 既然没有 Offset，倾斜完全由速度环 PID 产生。
        # 我们允许 0~20度的自然加速倾斜。超过 20度 视为 PID 参数过激，开始惩罚。
        pitch_deg_abs = torch.abs(self.lqr_controller.pitch)
        
        # 超过 25 度开始惩罚 (软红线)
        # 超过 30 度极速惩罚 (硬红线)
        danger_diff = torch.clamp(pitch_deg_abs - 25.0, min=0.0)
        return torch.square(danger_diff)

    def _reward_lqr_pitch_dot(self):
        # 抑制点头震荡
        return torch.square(self.lqr_controller.pitch_dot)

    # ========================================================================
    # 3. 动力学平滑性 (Dynamics Smoothness): 消除顿挫与震荡
    # ========================================================================
    def _reward_dof_acc(self):
        # [关键] 关节加速度惩罚
        # 防止 PID 参数(尤其是 D 项)突变引起电机高频震荡
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel)/self.dt), dim=1)

    def _reward_dof_vel(self):
        # 关节速度惩罚：防止腿部像划水一样疯转
        return torch.sum(torch.square(self.dof_vel[:, :4]), dim=1)

    def _reward_dof_force(self):
        # 力矩/能耗惩罚
        # 鼓励 RL 寻找“用最小力气达到目标速度”的 PID 组合
        return torch.sum(torch.square(self.dof_force), dim=1)

    def _reward_base_accel_x(self):
        # 机体加速度惩罚
        # 作用：防止 RL 突然把 Kp 拉满导致机器人像被踢了一脚一样顿挫起步
        acc = (self.base_lin_vel[:, 0] - self.last_base_lin_vel[:, 0]) / self.dt
        return torch.square(acc)

    # ========================================================================
    # 4. 参数调优约束 (Parameter Regularization): 针对 9维动作
    # ========================================================================
    def _reward_action_rate(self):
        # 参数平滑性：PID 参数绝不能突变
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_magnitude(self):
        # 参数正则化：鼓励参数回归默认值 (0)
        # 使用平方惩罚即可，因为现在没有 Offset 去锁边界了，PID 参数通常在 0 附近波动
        return torch.sum(torch.square(self.actions), dim=1)
        
    def _reward_collision(self):
        # 碰撞惩罚 (限制最大值，防止 Value Loss 爆炸)
        collision = torch.zeros(self.num_envs, device=self.device)
        for idx in self.reset_links:
            collision += torch.square(self.connect_force[:, idx, :]).sum(dim=1)
        return torch.clamp(collision, max=20.0) 
        
    def _reward_lin_vel_z(self):
        # 防止弹跳
        return torch.square(self.base_lin_vel[:, 2])
        
    def _reward_ang_vel_xy(self):
        # 防止晃动
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    # [新增] 停滞惩罚：指令让你走，你不走，就重罚
    def _reward_stall(self):
        # 1. 定义条件 (生成布尔张量)
        # 注意：必须使用位运算符 '&' 来代替 Python 的 'and'
        # 这里的 condition 是一个形状为 (num_envs,) 的由 0 和 1 组成的向量
        is_slow = torch.abs(self.base_lin_vel[:, 0]) < 0.3
        cmd_is_fast = self.commands[:, 0] > 0.5
        
        # 两个条件同时满足 (即正在堵转)
        stall_mask = is_slow & cmd_is_fast

        # 2. 计算误差 (对所有机器人先算一遍，不用担心浪费性能，GPU 很快)
        vel_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        
        # 3. 应用掩码 (过滤)
        # 如果 mask 是 True (1)，保留 error；如果是 False (0)，结果变成 0
        active_penalty = vel_error * stall_mask
        
        # 4. 返回平方
        return torch.square(active_penalty)
    
    def _reward_vel_overshoot_soft(self):
        # 取绝对值，只关心大小，不关心方向
        v_cmd_abs = torch.abs(self.commands[:, 0])
        v_real_abs = torch.abs(self.base_lin_vel[:, 0])
        
        # [核心逻辑] 定义“合法超调范围” (Tolerance)
        # 规则：允许 10% 的比例超调，或者至少 0.1 m/s 的绝对超调
        # 这样处理了 v_cmd = 0 的情况，也给了高速时更大的空间
        tolerance_ratio = 0.10  # 10%
        tolerance_abs = 0.10    # 0.1 m/s
        
        allowed_overshoot = torch.max(v_cmd_abs * tolerance_ratio, 
                                      torch.tensor(tolerance_abs, device=self.device))
        
        # 计算“违规超调量”
        # 只有当 实际速度 > (指令 + 豁免值) 时，diff 才是正数
        diff = v_real_abs - (v_cmd_abs + allowed_overshoot)
        
        # 只惩罚大于 0 的部分 (即违规部分)
        penalty = torch.clamp(diff, min=0.0)
        
        # 平方惩罚，对严重违规重拳出击
        return torch.square(penalty)