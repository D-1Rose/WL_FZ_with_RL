import copy
from controllers.Controller import BatchedLQRController
from controllers.Controller import BatchedVMC
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
                   robot_morphs="urdf", show_viewer=False,
                     device="cuda:0", train_mode=True):
        self.device = torch.device(device)

        self.mode = train_mode   #True训练模式开启
        
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_slice_obs = obs_cfg["num_slice_obs"]
        self.history_length = obs_cfg["history_length"]
        # self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  #输出维度，可自定义，这里暂时是2
        self.num_dofs = env_cfg["num_dofs"]  # 6
        self.num_commands = command_cfg["num_commands"]  # 4
        self.curriculum_cfg = curriculum_cfg
        self.domain_rand_cfg = domain_rand_cfg
        self.terrain_cfg = terrain_cfg
        self.num_respawn_points = self.terrain_cfg["num_respawn_points"]
        self.respawn_points = self.terrain_cfg["respawn_points"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.002  
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
        self.action_smooth_factor = 0.2  # 推荐 0.6~0.8。越大越平滑，反应越慢。
        self.common_step_counter = 0

        # [优雅重构] 日志相关的成员变量
        self.writer = None
        self.log_step = 0  # 内部计数器，不再依赖外部传入的 [0]


        # 修正：初始化腿部关节PID状态（形状为 [num_envs, 4]）
        self.pos_integral = torch.zeros((self.num_envs, 4), device=self.device)  # 原 [4] → [1,4]（num_envs=1）
        self.last_pos_error = torch.zeros((self.num_envs, 4), device=self.device)


        self.vel_integral = torch.zeros((self.num_envs, 2), device=self.device)  # 2个轮子
        self.last_vel_error = torch.zeros((self.num_envs, 2), device=self.device)

        self.count = 0

        

# =========== [插入开始] 1. 初始化 Teacher 的观测维度与数据 ===========
        # Teacher 观测 = Actor观测 + 物理真值(1+7维) + 地形扫描(35维)
        self.n_links = 7  # 请根据你的机器人实际情况修改这个数字！
        
        # Teacher 观测 = Actor观测(历史) + 物理真值 + 地形扫描(35)
        # 物理真值 = 摩擦力(1) + 全身质量(n_links)
        self.num_scan_points = 35
        self.num_physical_params = 1 + self.n_links  # 摩擦力(1) + 全身质量(7) = 8
        self.num_privileged_features = self.num_physical_params + self.num_scan_points
        self.num_privileged_obs = self.num_obs + self.num_privileged_features
        
        # 初始化 buffer (用于存放给 Critic 看的数据)
        self.privileged_obs_buf = torch.zeros(
            self.num_envs, self.num_privileged_obs, device=self.device, dtype=gs.tc_float
        )

        # 准备地形扫描网格 (5x7)
        y = torch.linspace(-0.5, 0.5, 5, device=self.device)
        x = torch.linspace(-1, 1, 7, device=self.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        self.scan_dots = torch.stack([x.flatten(), y.flatten()], dim=1) # (35, 2)

        # 初始化物理真值存储变量 (用于 domain_rand 存数据)
        self.stored_friction = torch.zeros(self.num_envs, 1, device=self.device)
        self.stored_mass = torch.zeros(self.num_envs, self.n_links, device=self.device)
        # =========== [插入结束] ===========

        


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
                # height_field = cv2.imread("assets/terrain/png/agent_eval_gym_d30_0.10.png", cv2.IMREAD_GRAYSCALE)
                self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                pos = (-0.1,-2.0,0.0),
                quat = (1,0,0,1),
                height_field = height_field,
                horizontal_scale=self.horizontal_scale, 
                vertical_scale=self.vertical_scale,
                ),)     
                # print("\033[1;34m respawn_points: \033[0m",self.base_init_pos)

                # self.terrain = self.scene.add_entity(
                # morph=gs.morphs.Mesh(
                # pos = (-1,-0.2,0.0),
                # quat = (0,0,0,1),
                # file = "/home/huang/wheel_leg/wheel_legged_genesis_new/assets/terrain/single_bridge.obj"
                # ),)     
        # add robot
        base_init_pos = self.base_init_pos.cpu().numpy()
        if self.terrain_cfg["terrain"]:
            if self.mode:
                base_init_pos = self.base_terrain_pos[0].cpu().numpy()

        match robot_morphs:
            case "urdf":
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file="assets/description/urdf/wheel_leg.urdf",
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
                        file="assets/description/urdf/wheel_leg.urdf",
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

        # ================== 参数注册表 (The Registry) ==================
        # 定义动作如何映射到参数，并增加 "clip" 字段进行安全截断
        self.param_registry = {
            # --- 1. 模糊控制参数 (线性映射) ---
            # [Clip 逻辑] 防止角度过大或系数越界
            "gamma_d_g": {"idx": 0, "default": 75*math.pi/180, "type": "linear", "scale": 30*math.pi/180, "clip": [45*math.pi/180, 89*math.pi/180]},#不能=90度，此处是奇点
            "eth":       {"idx": 1, "default": 0.06,            "type": "linear", "scale": 0.025,            "clip": [0.01, 0.1]},
            # "ld_max":    {"idx": 2, "default": 0.9,             "type": "linear", "scale": 0.4,             "clip": [0.6, 1.8]},
            # "ld_min":    {"idx": 3, "default": 0.3,             "type": "linear", "scale": 0.15,             "clip": [0, 0.6]},
            "a":         {"idx": 2, "default": 0.06,             "type": "linear", "scale": 0.05,             "clip": [0.01, 0.1]},
            # # --- 2. LQR PID 增益 (指数映射) ---
            # "pitch_kp":  {"idx": 5, "default": 20,  "type": "linear", "scale": 10, "clip": [5, 50]}, #1
            # # Pitch KD: 负责阻尼。
            # "pitch_kd":  {"idx": 6, "default": 0.13, "type": "linear", "scale": 0.065, "clip": [0.01, 0.13]},#0.015
            
            # # Vel KP: 负责动力。
            # "vel_kp":    {"idx": 7, "default": 4,  "type": "linear", "scale": 8.0, "clip": [0.0, 12.0]},#15
            # # Vel Ki: 积分项，防止过大导致发散
            # "vel_ki":    {"idx": 8, "default": 1,   "type": "linear", "scale": 5.0,  "clip": [0.0, 6.0]},#exponential 4.0
            # "vel_kd":    {"idx": 9, "default": 0.001,  "type": "linear", "scale": 1,    "clip": [0.00, 0.2] },#0.1
            # "yaw_dot_kp":    {"idx": 12, "default": 2.0,  "type": "linear", "scale": 3.0,        "clip": [0.0, 6.0]},#1.0

            # --- 3. VMC 参数 (地形泛化能力核心) ---
            # "vmc_z_l":   {"idx": 9,  "default": 0.0,  "type": "linear", "scale": 0.12, "clip": [-0.05, 0.12]},
            # "vmc_z_r":   {"idx": 10, "default": 0.0,  "type": "linear", "scale": 0.12, "clip": [-0.05, 0.12]},
            # "vmc_k":     {"idx": 9,  "default": 400.0,"type": "linear", "scale": 150,    "clip": [250.0, 650]},
            # "vmc_d":     {"idx": 10, "default": 40,  "type": "linear",     "scale": 20,  "clip": [25, 60]},
            # "vmc_b_l":  {"idx": 12, "default": 0.0,  "type": "linear", "scale": 0.02,    "clip": [-0.02, 0.02]},
            # "vmc_b_r":  {"idx": 13, "default": 0.0,  "type": "linear", "scale": 0.02,    "clip": [-0.02, 0.02]},
            # "vmc_roll_k": {"idx": 5, "default": 2500.0, "type": "linear", "scale": 0.0, "clip": [1000.0, 5000.0]},
            # "vmc_roll_d": {"idx": 6, "default": 100.0,  "type": "linear", "scale": 0,  "clip": [0.0, 300.0]},
        }

        # [新增] 初始化控制器参数结构
        # 告诉底层控制器："咱们以后就用这些参数名通讯"
        self.lqr_controller.init_params(self.param_registry)
        # self.vmc_controller.init_params(self.param_registry)

        self.height_target = torch.ones(self.num_envs, device=self.device) * 0.2779

        # 校验动作维度 (防止 config 写错)
        required_actions = max([p["idx"] for p in self.param_registry.values()]) + 1

        print(f"self.param_registry: {self.param_registry}")
        print(f"required_actions: {required_actions}")

        if self.num_actions < required_actions:
             print(f"⚠️  警告: param_registry 需要 {required_actions} 维动作，但 env_cfg 只有 {self.num_actions} 维。请修改 train.py!")


        # [新增] 打印动作映射表 (让你一眼便知)
        print("\n" + "="*40)
        print(f"🤖  RL 动作映射表 (Total: {self.num_actions} dim)")
        print("="*40)
        print(f"{'Index':<6} | {'Parameter Name':<20} | {'Type':<10}")
        print("-" * 40)
        
        # 按 idx 排序打印
        sorted_registry = sorted(self.param_registry.items(), key=lambda x: x[1]['idx'])
        for name, cfg in sorted_registry:
            print(f"{cfg['idx']:<6} | {name:<20} | {cfg['type']:<10}")
        print("="*40 + "\n")

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

        # self.slice_obs_buf = torch.zeros((self.num_envs, self.num_slice_obs), device=self.device, dtype=gs.tc_float)
        # self.history_obs_buf = torch.zeros((self.num_envs, self.history_length, self.num_slice_obs), device=self.device, dtype=gs.tc_float)
        # # [新增] 真实历史长缓存 (The Vault)
        # # 我们需要存过去 30 帧，哪怕只用其中 5 帧，这样才能回溯到 0.3秒前
        # self.true_history_len = 30
        # self.true_history_buf = torch.zeros(
        #             (self.num_envs, self.true_history_len, self.num_slice_obs), 
        #             device=self.device, dtype=gs.tc_float
        #         )
        # # [新增] 稀疏采样索引
        # # index 1=t-1, 3=t-3, ..., 24=t-24
        # self.history_indices = [1, 3, 6, 12, 24]
        # # 校验配置
        # if self.history_length != 5:#当前配置中history_length=5
        #      print(f"注意 : Config history_length={self.history_length}, 这表示选取几帧历史观测,在train配置中修改了history_length的话,需要增加选取哪几帧请在self.history_indices处修改选取的历史观测索引")
        
        # [替换为新代码]
        self.slice_obs_buf = torch.zeros((self.num_envs, self.num_slice_obs), device=self.device, dtype=gs.tc_float)
        
        # ================= [修改] 简单连续历史缓存 (FIFO 队列) =================
        # 只需要存最近的 5 帧 (history_length)
        # 形状: (N, 5, Dim)
        
        # 1. Actor 用的脏历史 (带噪声)
        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.history_length, self.num_slice_obs), 
            device=self.device, dtype=gs.tc_float
        )
        
        # 2. Critic 用的干净历史 (无噪声，上帝视角)
        self.clean_history_buf = torch.zeros_like(self.obs_history_buf)




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
        # if(self.env_cfg["termination_if_base_connect_plane_than"]&self.mode):
        #     self.reset_links = [(self.robot.get_link(name).idx_local) for name in self.env_cfg["connect_plane_links"]]
        
        # [修复] 始终初始化 reset_links，因为碰撞惩罚(_reward_collision)也需要用到它
        # 即使不开启触地重置，我们依然需要知道哪些 link 是"敏感部位"
        self.reset_links = [(self.robot.get_link(name).idx_local) for name in self.env_cfg["connect_plane_links"]]

        #跪地重启逻辑在 check_termination 里会再次判断 flag，所以这里拿出来是安全的 
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

        # print("self.init_dof_pos",self.init_dof_pos)
        #初始化角度
        self.reset()
        


    def _resample_commands(self, envs_idx):
            """
            指令重采样：向量化版本
            """
            # [新增] 0. 极速防御：如果没有环境需要重置，直接返回！
            # 这就是报错的根源：len(envs_idx) 可能为 0
            if len(envs_idx) == 0:
                return
            # --------------------------------------------------------
            # A. 采样 Linear Velocity X (线速度) - 融合分桶逻辑
            # --------------------------------------------------------
            x_min = self.command_ranges[envs_idx, 0, 0]
            x_max = self.command_ranges[envs_idx, 0, 1]
            
            # 默认使用均匀采样作为底板
            self.commands[envs_idx, 0] = gs_rand_float(x_min, x_max, (len(envs_idx),), self.device)

            # [可选] 如果配置了分桶，覆盖上面的均匀采样
            if "distribution_buckets" in self.command_cfg:
                buckets = self.command_cfg["distribution_buckets"]
                probs = torch.tensor([b["prob"] for b in buckets], device=self.device)
                # 随机选桶
                bucket_indices = torch.multinomial(probs, len(envs_idx), replacement=True)
                
                x_span = x_max - x_min
                for i, bucket in enumerate(buckets):
                    mask = (bucket_indices == i)
                    if not mask.any(): continue
                    
                    ids = envs_idx[mask]
                    pct_min, pct_max = bucket["range_pct"]
                    # 计算分桶后的实际范围
                    v_min = x_min[mask] + pct_min * x_span[mask]
                    v_max = x_min[mask] + pct_max * x_span[mask]
                    
                    self.commands[ids, 0] = gs_rand_float(v_min, v_max, (len(ids),), self.device)

            # --------------------------------------------------------
            # B. 采样 Angular Velocity (角速度) - [完美复刻你的物理约束]
            # --------------------------------------------------------
            # 原逻辑：ang_vel_max = (wheel_max_w*R*2 - |v|*2) / spacing
            # 也就是：allowed_w = (wheel_max_v - |v_lin|) * 2 / spacing
            
            if self.command_cfg.get("limit_cmd_random", False):
                # 1. 获取参数
                R = self.command_cfg["wheel_radius"]
                max_w = self.command_cfg["wheel_max_w"]
                spacing = self.command_cfg["wheel_spacing"]
                
                # 2. 获取刚才生成的线速度 (绝对值)
                v_lin = torch.abs(self.commands[envs_idx, 0])
                
                # 3. 向量化计算最大允许角速度
                # clamp(min=0) 防止线速度过大时出现负数
                wheel_v_limit = max_w * R
                allowed_w = (wheel_v_limit - v_lin).clamp(min=0.0) * 2.0 / spacing
                
                # 4. 压缩范围
                # 你的原逻辑：ang_vel_low.clamp_min_(-ang_vel_max)
                # 这里我们先取基础范围，再和物理极限取交集
                raw_low = self.command_ranges[envs_idx, 2, 0]
                raw_high = self.command_ranges[envs_idx, 2, 1]
                
                # 确保 low 不低于 -allowed_w，high 不高于 allowed_w
                final_low = torch.max(raw_low, -allowed_w)
                final_high = torch.min(raw_high, allowed_w)
                
                # 5. 采样
                self.commands[envs_idx, 2] = gs_rand_float(final_low, final_high, (len(envs_idx),), self.device)
                
                # 6. 其他通道处理 (复刻你的逻辑)
                # Y轴速度采样
                self.commands[envs_idx, 1] = gs_rand_float(
                    self.command_ranges[envs_idx, 1, 0],
                    self.command_ranges[envs_idx, 1, 1], 
                    (len(envs_idx),), self.device
                )
                # 高度采样 (如果存在)
                if self.command_ranges.shape[1] > 3:
                    self.commands[envs_idx, 3] = gs_rand_float(
                        self.command_ranges[envs_idx, 3, 0], 
                        self.command_ranges[envs_idx, 3, 1], 
                        (len(envs_idx),), self.device
                    )
                
            else:
                # 不开 limit_cmd_random 时的简单逻辑
                for cmd_i in range(self.num_commands):
                    self.commands[envs_idx, cmd_i] = gs_rand_float(
                        self.command_ranges[envs_idx, cmd_i, 0],
                        self.command_ranges[envs_idx, cmd_i, 1],
                        (len(envs_idx),), self.device
                    )

            # 最后强制 Y 轴清零 (复刻你的代码最后一行)
            self.commands[envs_idx, 1] = 0.

    def set_commands(self,envs_idx,commands):
        self.commands[envs_idx]=torch.tensor(commands,device=self.device, dtype=gs.tc_float)

# ================= [新增] 优雅的日志设置接口 =================
    def set_logger(self, writer):
        """外部脚本调用此函数注入 Writer"""
        self.writer = writer
        self.log_step = 0
        print(f"✅ 环境日志系统已就绪，当前 Log Step: {self.log_step}")

    # ================= [新增] 私有日志记录函数 =================
    def _log_metrics(self):
        """专门负责向 TensorBoard 写入数据，不污染 step 函数的主逻辑"""
        if self.writer is None:
            return

        # 1. 记录模糊参数
        lqr_params = self.lqr_controller.params
        if 'gamma_d_g' in lqr_params:
            self.writer.add_scalar('Params/Fuzzy_Gamma', lqr_params['gamma_d_g'][0], self.log_step)
            self.writer.add_scalar('Params/Fuzzy_Eth', lqr_params['eth'][0], self.log_step)
            # self.writer.add_scalar('Params/Fuzzy_Ld_Max', lqr_params['ld_max'][0], self.log_step)
            # self.writer.add_scalar('Params/Fuzzy_Ld_Min', lqr_params['ld_min'][0], self.log_step)
            self.writer.add_scalar('Params/Fuzzy_A', lqr_params['a'][0], self.log_step)

        # # 2. 记录 VMC 参数
        # vmc_params = self.vmc_controller.params
        # if 'vmc_roll_k' in vmc_params:
        #     self.writer.add_scalar('Params/VMC_Roll_K', vmc_params['vmc_roll_k'][0], self.log_step)
        #     self.writer.add_scalar('Params/VMC_Roll_D', vmc_params['vmc_roll_d'][0], self.log_step)


        # self.writer.add_scalar('Dofs/L1', self.dof_pos[0, 0], self.log_step)
        # self.writer.add_scalar('Dofs/L2', self.dof_pos[0, 1], self.log_step)
        # self.writer.add_scalar('Dofs/R1', self.dof_pos[0, 2], self.log_step)
        # self.writer.add_scalar('Dofs/R2', self.dof_pos[0, 3], self.log_step)
        # self.writer.add_scalar('Dofs/L3', self.dof_pos[0, 4], self.log_step)
        # self.writer.add_scalar('Dofs/R3', self.dof_pos[0, 5], self.log_step)

        # self.writer.add_scalar('Dofs/L1_Vel', self.dof_vel[0, 0], self.log_step)
        # self.writer.add_scalar('Dofs/L3_Vel', self.dof_vel[0, 2], self.log_step)


        # 3. 记录跟踪误差
        self.writer.add_scalar('Tracking/Vel_X_Error', self.commands[0,0] - self.base_lin_vel[0,0], self.log_step)
        self.writer.add_scalar('Tracking/Command_Vel', self.commands[0,0], self.log_step)
        self.writer.add_scalar('Tracking/Vel_X', self.base_lin_vel[0,0], self.log_step)
        self.writer.add_scalar('Tracking/Height_Real', self.base_pos[0,2], self.log_step)

        # 4. 记录姿态角 (度)
        # [修复] 应该取第0个环境的 [0], [1], [2] 分量
        # self.base_euler[0] 是一个 [3] 的向量 (r, p, y)
        # degrees = self.base_euler[0] # * torch.pi / 180.0
        # r = degrees[0]
        # p = degrees[1]
        # y = degrees[2]

        env_id = 0
        r = self.base_euler[env_id, 0]
        p = self.base_euler[env_id, 1]
        y = self.base_euler[env_id, 2]
        p_com = self.lqr_controller.pitch_com[0]
        p_vel = self.base_ang_vel[env_id, 1]
        self.writer.add_scalar('Attitude/Roll', r, self.log_step)
        self.writer.add_scalar('Attitude/Pitch', p, self.log_step)
        self.writer.add_scalar('Attitude/Yaw', y, self.log_step)
        self.writer.add_scalar('Attitude/Pitch_Vel', p_vel, self.log_step)
        self.writer.add_scalar('Attitude/Pitch_Com', p_com, self.log_step)                



        # 计数器自增
        self.log_step += 1

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

    # ================== [新增] 自动化参数解析与更新 ==================
        new_params = {}
        
        # 遍历注册表，自动计算所有参数
        for name, cfg in self.param_registry.items():
            idx = cfg["idx"]
            
            # 保护机制：防止动作维度不够导致越界报错
            # if idx >= real_actions.shape[1]:
            #     continue
                
            raw_action = real_actions[:, idx]
            default_val = cfg["default"]
            scale_val = cfg["scale"]
            
            # 1. 根据类型计算值
            if cfg["type"] == "linear":
                # 线性映射: Val = Default + Action * Scale
                # 适合: 角度、长度、偏移量
                val = default_val + raw_action * scale_val

            elif cfg["type"] == "exponential":
                # 指数映射: Val = Default * (Scale ^ Action)
                # 适合: Kp, Kd, Ki, 刚度 K 等必须 > 0 且跨度大的参数
                # action=0 -> 1倍; action=1 -> Scale倍; action=-1 -> 1/Scale倍
                val = default_val * torch.pow(torch.tensor(scale_val, device=self.device), raw_action)
            
            else:
                # 默认情况 (不应发生)
                val = torch.full_like(raw_action, default_val)
            
            # 2. 安全范围截断 (Clip)
            if "clip" in cfg:
                low, high = cfg["clip"]
                val = torch.clamp(val, low, high)

                # --- Curriculum 介入：Stage 1 冻结 Bias ---

                
            new_params[name] = val

        # ================== [核心] 一键更新所有控制器 ==================
        # 这一步把算好的参数字典直接灌入控制器，替代了以前的 set_fuzzy_params
        self.lqr_controller.update_params(new_params)
        # self.vmc_controller.update_params(new_params)
        
        self.lqr_controller.set_commands(
            velocity_d = self.commands[:, 0],
            yaw_d_dot = self.commands[:, 2],
        )
        self.vmc_controller.set_commands(
            height_target = self.height_target #self.commands[:, 3]  # 把 command 的第4维(高度)传进去
        )


# ===================================================================
        # [核心修改] 传感器模拟层：提前加噪，然后喂给控制器
        # ===================================================================
        
        # (A) 获取物理真值 (Ground Truth) -- 这是上帝视角的绝对真理
        gt_dof_pos = self.robot.get_dofs_position()
        gt_dof_vel = self.robot.get_dofs_velocity()
        
        # 获取基座真值 (建议重新获取一次，确保是当前时刻最新的)
        gt_quat = self.robot.get_quat()
        gt_ang_vel = self.robot.get_ang()
        inv_quat_rot = inv_quat(gt_quat)
        
        # 转换到基座坐标系 (Body Frame)
        gt_base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_quat_rot)
        gt_base_ang_vel = transform_by_quat(gt_ang_vel, inv_quat_rot)
        gt_base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(gt_quat) * self.inv_base_init_quat, gt_quat))

        # (B) 制造“脏数据” (模拟真实传感器)
        if self.noise["use"]:
            # 噪声
            noisy_dof_pos = gt_dof_pos + torch.randn_like(gt_dof_pos) * self.noise["dof_pos"][0] + (torch.rand_like(gt_dof_pos)*2-1) * self.noise["dof_pos"][1]
            noisy_dof_vel = gt_dof_vel + torch.randn_like(gt_dof_vel) * self.noise["dof_vel"][0] + (torch.rand_like(gt_dof_vel)*2-1) * self.noise["dof_vel"][1]
            noisy_base_ang_vel = gt_base_ang_vel + torch.randn_like(gt_base_ang_vel) * self.noise["ang_vel"][0] + (torch.rand_like(gt_base_ang_vel)*2-1) * self.noise["ang_vel"][1]
            noisy_base_euler = gt_base_euler + torch.randn_like(gt_base_euler) * self.noise["base_euler"][0] + (torch.rand_like(gt_base_euler)*2-1) * self.noise["base_euler"][1]
            self.projected_gravity += torch.randn_like(self.projected_gravity) * self.noise["gravity"][0] + (torch.rand_like(self.projected_gravity)*2-1) * self.noise["gravity"][1]


        else:
            # 如果关闭噪声，则直接透传真值
            noisy_dof_pos = gt_dof_pos
            noisy_dof_vel = gt_dof_vel
            noisy_base_ang_vel = gt_base_ang_vel
            noisy_base_euler = gt_base_euler

        # (C) [关键一步] 把“脏数据”丢给控制器
        # 这样 LQR 算出来的力矩才是真实且鲁棒的！
        self.lqr_controller.set_state(
            noisy_dof_pos, 
            noisy_dof_vel, 
            gt_base_lin_vel,    # 线速度通常比较难观测，LQR里一般不用，或者用脏的也可以
            noisy_base_euler, 
            noisy_base_ang_vel
        )
        self.vmc_controller.set_state(
            noisy_dof_pos, 
            noisy_dof_vel, 
            gt_base_lin_vel, 
            noisy_base_euler, 
            noisy_base_ang_vel
        )


        # self.base_quat[:] = self.robot.get_quat()
        # self.base_euler = quat_to_xyz(
        #     transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        # )
        # # =============  统一获取所有状态（只调用一次！） =============
        # all_dof_pos = self.robot.get_dofs_position()  # [num_envs, n_dofs]
        # all_dof_vel = self.robot.get_dofs_velocity()  # [num_envs, n_dofs]
        # base_vel = self.base_lin_vel              # [num_envs, 3]
        # base_quat = self.robot.get_quat()             # [num_envs, 4]
        # base_euler = self.base_euler               # [num_envs, 3]
        # base_ang = self.base_ang_vel               # [num_envs, 3]
        # inv_base_quat = inv_quat(self.base_quat)
        
        # # =============  传递状态给控制器 =============
        # self.lqr_controller.set_state(all_dof_pos, all_dof_vel, base_vel, base_euler, base_ang)
        # self.vmc_controller.set_state(all_dof_pos, all_dof_vel, base_vel, base_euler, base_ang)


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

        # -------------------------------------------------------------
        # 步骤 1: 备份真值 (给 Critic 用的上帝视角)
        # -------------------------------------------------------------
        # 必须在加噪声之前执行！
        clean_base_lin_vel = self.base_lin_vel.clone()
        clean_base_ang_vel = self.base_ang_vel.clone()
        clean_dof_pos = self.dof_pos.clone()
        clean_dof_vel = self.dof_vel.clone()
        clean_base_euler = self.base_euler.clone()

        if self.noise["use"]:
            self.base_ang_vel[:] += torch.randn_like(self.base_ang_vel) * self.noise["ang_vel"][0] + (torch.rand_like(self.base_ang_vel)*2-1) * self.noise["ang_vel"][1]
            self.projected_gravity += torch.randn_like(self.projected_gravity) * self.noise["gravity"][0] + (torch.rand_like(self.projected_gravity)*2-1) * self.noise["gravity"][1]
            self.dof_pos[:] += torch.randn_like(self.dof_pos) * self.noise["dof_pos"][0] + (torch.rand_like(self.dof_pos)*2-1) * self.noise["dof_pos"][1]
            self.dof_vel[:] += torch.randn_like(self.dof_vel) * self.noise["dof_vel"][0] + (torch.rand_like(self.dof_vel)*2-1) * self.noise["dof_vel"][1]
            self.base_euler[:] += torch.randn_like(self.base_ang_vel) * self.noise["base_euler"][0] + (torch.rand_like(self.base_ang_vel)*2-1) * self.noise["base_euler"][1]

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
    # A. 准备公共数据
        lqr_params = self.lqr_controller.current_fuzzy_params 
        # vmc_params = self.vmc_controller.current_vmc_params   
        # all_phy_params = torch.cat([lqr_params, vmc_params], dim=-1)
        all_phy_params = lqr_params
        
        # B. 构建 Actor 的当前观测 (使用被污染的 self.xxx)
        # 注意：LQR 的 pitch 也是脏的，因为它依赖脏的 IMU 数据计算
        dirty_pitch_obs = torch.cat([
            self.lqr_controller.pitch.unsqueeze(-1),
            self.lqr_controller.pitch_dot.unsqueeze(-1),
        ], axis=-1)

        # self.count += 1
        # if self.count % 150 == 0:
        #     print("=== Step 函数观测调试信息 ===")
        #     print("self.base_lin_vel: ",self.base_lin_vel[0,:])
        #     print("self.base_ang_vel: ",self.base_ang_vel[0,:])
        #     print(f"self.projected_gravity: {self.projected_gravity[0,:]}")
        #     print(f"self.base_euler: {self.base_euler[0,:]}")
        #     print("self.commands: ",{self.commands * self.commands_scale})
        #     print(f"self.obs_dof_pos: {self.dof_pos[:,0:4] - self.default_dof_pos[:,0:4]}")
        #     print(f"self.dof_pos: {self.dof_pos[0,:]}")
        #     print(f"self.default_dof_pos: {self.default_dof_pos[0,:]}")
        #     print(f"self.dof_vel: {self.dof_vel[0,:]}")
        #     print("self.actions: ",self.actions[0,:])
        #     print("all_phy_params: ",all_phy_params[0,:])
        #     print(f"dirty_pitch_obs: {dirty_pitch_obs[0,:]}")
        #     print("========= 观测结束 ========")

        self.slice_obs_buf = torch.cat([
            self.base_lin_vel ,   # 脏
            self.base_ang_vel ,   # 脏
            self.projected_gravity,
            self.commands ,
            (self.dof_pos[:,0:4] - self.default_dof_pos[:,0:4]), # 脏
            self.dof_vel , #到此23维       # 脏
            self.actions ,
            all_phy_params , 
            dirty_pitch_obs,                                  # 脏
        ], axis=-1)

        # C. 构建 Critic 的当前观测 (使用 clean_xxx)
        # [修正] 直接使用 clean_base_ang_vel 的 Y 分量作为 Pitch Dot 真值
        # 这样就和 Actor 的观测逻辑（IMU Y轴）完美对应了，不需要额外的 cos 变换
        clean_pitch_obs = torch.cat([
            clean_base_euler[:, 1].unsqueeze(-1),   # 净 Pitch
            clean_base_ang_vel[:, 1].unsqueeze(-1)  # 净 Pitch Dot
        ], axis=-1)

        clean_slice_obs = torch.cat([
            clean_base_lin_vel,# * self.obs_scales["lin_vel"],  # 净
            clean_base_ang_vel, #* self.obs_scales["ang_vel"],  # 净
            self.projected_gravity,
            self.commands,# * self.commands_scale,
            (clean_dof_pos[:,0:4] - self.default_dof_pos[:,0:4]),# * self.obs_scales["dof_pos"], # 净
            clean_dof_vel, #* self.obs_scales["dof_vel"],       # 净
            self.actions,
            all_phy_params,
            clean_pitch_obs                                   # 净
        ], axis=-1)


        # self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
# 逻辑：obs = [历史(t-N...t-1), 当前(t)]
        # 时序：索引 0 是最旧的，索引 -1 是最新的 (Forward Time)
        
        # 1. 【生成最终观测】 (History + Current)
        # Actor 观测 (脏)
        # cat([ (N, 5, D), (N, 1, D) ]) -> (N, 6, D) -> flatten
        obs_seq = torch.cat([self.obs_history_buf, self.slice_obs_buf.unsqueeze(1)], dim=1)
        self.obs_buf = obs_seq.view(self.num_envs, -1)
        
        # Critic 观测 (净)
        clean_obs_seq = torch.cat([self.clean_history_buf, clean_slice_obs.unsqueeze(1)], dim=1)
        clean_obs_full = clean_obs_seq.view(self.num_envs, -1)

        # 2. 【更新历史缓存】 (左移淘汰最旧帧)
        # 只有当历史长度 > 0 时才需要更新
        if self.history_length > 0:
            # 整体左移: [t-5, t-4, t-3, t-2, t-1] -> [t-4, t-3, t-2, t-1, t-1]
            # 也就是丢弃最左边的(最旧的)
            self.obs_history_buf[:, :-1, :] = self.obs_history_buf[:, 1:, :].clone()
            self.obs_history_buf[:, -1, :] = self.slice_obs_buf # 最右边补入当前帧
            
            # Critic 同理
            self.clean_history_buf[:, :-1, :] = self.clean_history_buf[:, 1:, :].clone()
            self.clean_history_buf[:, -1, :] = clean_slice_obs

        # 3. 生成特权观测
        self.privileged_obs_buf = self._get_privileged_obs(clean_obs_full)
        
        # 4. 记录与返回
        self._log_metrics()


        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras # [重要] 第二个返回值改为 self.privileged_obs_buf


    def _get_privileged_obs(self, clean_obs=None):
        # 1. 兼容性处理
        # 如果调用时没传参（比如在 reset 或其他地方偶发调用），暂时用 obs_buf 顶替
        # 但在 step 函数里，我们一定会传 clean_obs_full 进来
        if clean_obs is None:
            clean_obs = self.obs_buf

        # 2. 获取地形扫描
        scan = self._get_height_scan()
        
        # 3. 获取真实的物理真值
        friction = self.stored_friction  # (N, 1)
        mass = self.stored_mass          # (N, 7)
        
        # 4. [关键修正] 拼接: [干净观测, 摩擦, 质量, 扫描]
        # 这里的 clean_obs 就是你 step 函数里传进来的 clean_obs_full (全真值)
        # 而不再是 self.obs_buf (脏数据)
        return torch.cat([clean_obs, friction, mass, scan], dim=-1)


    def _get_height_scan(self):
        # 1. 计算世界坐标
        count = 0
        yaw = self.base_euler[:, 2]
        base_x = self.base_pos[:, 0]
        base_y = self.base_pos[:, 1]
        
        c, s = torch.cos(yaw), torch.sin(yaw)
        x_local, y_local = self.scan_dots[:, 0], self.scan_dots[:, 1]
        
        x_world = x_local * c.unsqueeze(1) - y_local * s.unsqueeze(1) + base_x.unsqueeze(1)
        y_world = x_local * s.unsqueeze(1) + y_local * c.unsqueeze(1) + base_y.unsqueeze(1)
        
        # 2. [修复] 使用正确的变量名 self.terrain_height
        if hasattr(self, 'terrain_height') and self.terrain_height is not None:
            # 坐标 -> 索引
            x_idx = (x_world / self.horizontal_scale).long()
            y_idx = (y_world / self.horizontal_scale).long()
            
            # 边界保护
            rows, cols = self.terrain_height.shape
            x_idx = torch.clamp(x_idx, 0, cols - 1)
            y_idx = torch.clamp(y_idx, 0, rows - 1)
            
            # 查表 (terrain_height 已经是米为单位的高度了)
            heights = self.terrain_height[y_idx, x_idx]
        else:
            heights = torch.zeros_like(x_world)

        # 3. 相对高度
        scan = torch.clamp(heights - self.base_pos[:, 2].unsqueeze(1), -1.0, 1.0)
        
        # ==========================================
        # 🧪 [测试代码] 在这里打印，看看有没有数值！
        # ==========================================
        # 为了防止刷屏，只在第0个环境、每100步打印一次
        # if  count % 10 == 0:
        #      # 打印扫描的最大值和最小值
        #      # 如果全是 0.000，说明没扫到；如果有波动（比如 -0.15, 0.05），说明扫到了
        #      print(f"🔍 [地形测试] Scan Max: {scan.max().item():.3f}, Min: {scan.min().item():.3f}")
        # count += 1
        return scan
    

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf
    
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
        self.obs_history_buf[envs_idx] = 0.0
        self.clean_history_buf[envs_idx] = 0.0

        
        # 重置模糊参数为默认值
        # 直接调用重置接口
        self.lqr_controller.reset_params(envs_idx)
        # self.vmc_controller.reset_params(envs_idx)


        # reset buffers
        # [修改] 重置动作 Buffer
        self.actions[envs_idx] = 0.0      # 重置平滑器
        self.last_actions[envs_idx] = 0.0 # 重置延迟队列
        self.last_dof_vel[envs_idx] = 0.0
        self.reset_buf[envs_idx] = True
        # [这里] 现在才清空步数，逻辑就对了！
        self.episode_length_buf[envs_idx] = 0 
        self.episode_lengths[envs_idx] = 0.0


        self._resample_commands(envs_idx)  
        if self.mode:
            self.domain_rand(envs_idx)





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

        friction_ratio = self.friction_ratio_low + self.friction_ratio_range * torch.rand(len(envs_idx), self.robot.n_links, device=self.device)
        self.robot.set_friction_ratio(friction_ratio=friction_ratio,
                                      link_indices=np.arange(0, self.robot.n_links),
                                      envs_idx = envs_idx)
        # =========== [插入] 保存摩擦力真值 ===========
        # 取平均值作为代表，存下来给 Teacher 看
        self.stored_friction[envs_idx] = friction_ratio.mean(dim=1, keepdim=True)

        base_mass_shift = (self.base_mass_low + self.base_mass_range * torch.rand(len(envs_idx), 1, device=self.device))
        other_mass_shift =(-self.other_mass_low + self.other_mass_range * torch.rand(len(envs_idx), self.robot.n_links - 1, device=self.device))
        mass_shift = torch.cat((base_mass_shift, other_mass_shift), dim=1)
        self.robot.set_mass_shift(mass_shift=mass_shift,
                                  link_indices=np.arange(0, self.robot.n_links),
                                  envs_idx = envs_idx)
        # 现在改成保存完整的 mass_shift
        self.stored_mass[envs_idx] = mass_shift


        base_com_shift = (-self.domain_rand_cfg["random_base_com_shift"] / 2 + self.domain_rand_cfg["random_base_com_shift"] * torch.rand(len(envs_idx), 1, 3, device=self.device))
        other_com_shift = (-self.domain_rand_cfg["random_other_com_shift"] / 2 + self.domain_rand_cfg["random_other_com_shift"] * torch.rand(len(envs_idx), self.robot.n_links - 1, 3, device=self.device))
        com_shift = torch.cat((base_com_shift, other_com_shift), dim=1)
        self.robot.set_COM_shift(com_shift=com_shift,
                                 link_indices=np.arange(0, self.robot.n_links),
                                 envs_idx = envs_idx)

        # print(f"kp_low: {self.kp_low}, kp_range:{self.kp_range}, envs_idx:{envs_idx}")
        kp_shift = (self.kp_low + self.kp_range * torch.rand(len(envs_idx), self.num_dofs)) * self.kp[0]
        # [新增] 安全锁：强制将腿部关节 (0-3) 的底层 KP 设为 0，防止影响自己控制器的pid
        kp_shift[:, 0:4] = 0.0
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

    # =======================================================================================================================
    # [重新设计的奖励函数]
    # =======================================================================================================================
    

    def _reward_tracking_lin_vel(self):
        # 线速度跟踪：这是唯一的“正分”来源，必须强势
        # Sigma = 0.15 意味着误差 > 0.15m/s 时奖励迅速衰减，逼迫 RL 精细调参
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_lin_sigma"])

    def _reward_tracking_ang_vel(self):
        # 角速度跟踪
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_ang_sigma"])
    
    def _reward_tracking_roll(self):
        roll_error = torch.square( - self.base_euler[:, 0])
        return torch.exp(-roll_error / self.reward_cfg["tracking_roll_sigma"])
    
    def _reward_tracking_pitch(self):
        # 防止过度倾斜
        pitch_error = torch.square( - self.base_euler[:, 1])
        return torch.exp(-pitch_error / self.reward_cfg["tracking_pitch_sigma"])

    def _reward_action_rate(self):
        # 参数平滑性：希望 (last - current) 接近 0
        # 1. 先计算所有动作维度的平方差之和
        delta = self.last_actions - self.actions
        # dim=1 求和，得到每个环境的总变动量平方
        sq_diff = torch.sum(torch.square(delta), dim=1) 
        # 2. 放入高斯核
        return torch.exp(-sq_diff / self.reward_cfg["action_rate_sigma"])

    def _reward_action_magnitude(self):
        # 参数正则化：希望 actions 接近 0
        # dim=1 求和
        sq_mag = torch.sum(torch.square(self.actions), dim=1)
        return torch.exp(-sq_mag / self.reward_cfg["action_magnitude_sigma"])











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
        return torch.square(self.lqr_controller.pitch_dot/self.dt)
    
    def _reward_pitch_vel(self):
        # 机体俯仰角速度惩罚
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
    
    def _reward_ang_vel_xy_dot(self):
        ang_vel_dot = (self.base_ang_vel[:, :2] - self.last_base_ang_vel[:, :2]) / self.dt
        return torch.sum(torch.square(ang_vel_dot), dim=1)
    
    # [新增] 停滞惩罚：指令让你走，你不走，就重罚
    def _reward_stall(self):
        # 1. 定义条件 (生成布尔张量)
        # 注意：必须使用位运算符 '&' 来代替 Python 的 'and'
        # 这里的 condition 是一个形状为 (num_envs,) 的由 0 和 1 组成的向量
        cmd_is_fast = self.commands[:, 0] == 0
        vel_is_slow1 = torch.abs(self.base_lin_vel[:, 0]) <= 0.4
        vel_is_slow2 = 0.1 < torch.abs(self.base_lin_vel[:, 0]) 
        is_slow = torch.abs(self.lqr_controller.pitch_dot) > 0.05 

        
        # 两个条件同时满足 (即正在堵转)
        stall_mask = is_slow & cmd_is_fast & vel_is_slow1 & vel_is_slow2

        # 2. 计算误差 (对所有机器人先算一遍，不用担心浪费性能，GPU 很快)
        vel_error = torch.abs(self.lqr_controller.pitch_dot)
        
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
    

