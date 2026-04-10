import copy
from controllers.Pitch_controller import BatchedLQRController
from controllers.Pitch_controller import BatchedVMC
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

        # [核心修改] 时间尺度分离
        self.dt = 0.002              # 底层物理引擎和控制器的步长 (500Hz)
        self.decimation = 10         # 降频系数
        self.rl_dt = self.dt * self.decimation  # RL 策略的实际步长 (0.02s, 50Hz)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.rl_dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg  





        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.noise = obs_cfg["noise"]
        
        # [新增] 初始化动作平滑 Buffer
        self.smooth_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.action_smooth_factor = 0.5  # 平滑系数：0.65足够平滑但保持快速响应（不要太大）
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
        self.num_scan_points = 9
        self.num_physical_params = 1 + self.n_links  # 摩擦力(1) + 全身质量(7) = 8
        self.num_privileged_features = self.num_physical_params + self.num_scan_points
        self.num_privileged_obs = self.num_obs + self.num_privileged_features
        
        # 初始化 buffer (用于存放给 Critic 看的数据)
        self.privileged_obs_buf = torch.zeros(
            self.num_envs, self.num_privileged_obs, device=self.device, dtype=gs.tc_float
        )

        # 准备地形扫描网格 (3x3)
        y = torch.linspace(-0.1, 0.1, 3, device=self.device)
        x = torch.linspace(-0.1, 0.1, 3, device=self.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        self.scan_dots = torch.stack([x.flatten(), y.flatten()], dim=1) # (35, 2)

        # 初始化物理真值存储变量 (用于 domain_rand 存数据)
        self.stored_friction = torch.zeros(self.num_envs, 1, device=self.device)
        self.stored_mass = torch.zeros(self.num_envs, self.n_links, device=self.device)
        # =========== [插入结束] ===========

        
        # 定义双向终点线 (基于世界绝对坐标)
        # 地形从 X=-1.0 开始，到 X=12.5 左右结束
        self.finish_line_x_start = -2  # 倒车冲线坐标
        self.finish_line_x_end = 8    # 前进冲线坐标
        
        self.max_episode_length = 750   # 兜底超时

        self.theta_pitch = torch.zeros((self.num_envs,), device=self.device)  # 存储theta_pitch，供奖励函数使用

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                # max_FPS=int(0.5 / self.dt),
                max_FPS=1000,
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

        if self.terrain_cfg["terrain"]:
            print("\033[1;35m [Unified] Loading Terrain... \033[0m")
            
            # 1. 统一加载训练用地形图
            self.height_field = cv2.imread("assets/terrain/png/" + self.terrain_cfg["train"] + ".png", cv2.IMREAD_GRAYSCALE)
            self.terrain_height = torch.tensor(self.height_field, device=self.device) * self.vertical_scale
            
            # 2. 统一地形在世界坐标系中的位姿 (完全对齐你之前的 eval 偏移)
            self.terrain_pos = (0.0, 1.0, 0.0) 
            self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    pos=self.terrain_pos,
                    quat=(1.0, 0.0, 0.0, -1.0),
                    height_field=self.height_field,
                    horizontal_scale=self.horizontal_scale, 
                    vertical_scale=self.vertical_scale,
                ),
            )
            
            # 3. 将相对重置点转换为世界绝对坐标
            # 这样 reset_idx 就不再需要根据 mode 分支计算坐标
            self.base_terrain_pos = torch.zeros((self.num_respawn_points, 3), device=self.device)
            # terrain_origin = torch.tensor(self.terrain_pos, device=self.device)
            
            for i in range(self.num_respawn_points):
                # 绝对坐标 = 地形原点 + 配置中的偏移量 + 机器人初始站立高度
                offset = torch.tensor(self.respawn_points[i], device=self.device)
                self.base_terrain_pos[i] = offset + torch.tensor([0.0, 0.0, self.base_init_pos[2].item()], device=self.device)
            
            base_init_pos = self.base_terrain_pos[0].cpu().numpy()
        else:
            base_init_pos = self.base_init_pos.cpu().numpy()



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




        self.lqr_controller = BatchedLQRController(num_envs=self.num_envs,device=self.device)
        self.vmc_controller = BatchedVMC(num_envs=self.num_envs, device=self.device)

        # ================== 参数注册表 (The Registry) ==================
        # 定义动作如何映射到参数，并增加 "clip" 字段进行安全截断
        self.param_registry = {
# [核心修改] 将上限死死卡在 60 度！不允许 RL 构造极端陡峭的斜率
            "gamma_d_g": {"idx": 0, "default": 45*math.pi/180, "type": "linear", "scale": 15*math.pi/180, "clip": [15*math.pi/180, 60*math.pi/180]},
            # 提高 eth 的下限，强迫系统必须拥有至少 2% 的死区保护！
            "eth":       {"idx": 1, "default": 0.06,           "type": "linear", "scale": 0.05,            "clip": [0.02, 0.15]},
            "a":         {"idx": 2, "default": 0.01,            "type": "linear", "scale": 0.5,            "clip": [0.01, 0.99]},
            "k_out":     {"idx": 3, "default": 1.0,            "type": "linear", "scale": 0.25,             "clip": [0.5, 2.0]},
            # # Vel KP: 负责动力。
            # "vel_kp":    {"idx": 0, "default": 4,  "type": "linear", "scale": 8.0, "clip": [0.0, 12.0]},#15
            # Vel Ki: 积分项，防止过大导致发散
            # "vel_ki":    {"idx": 1, "default": 1,   "type": "linear", "scale": 5.0,  "clip": [0.0, 6.0]},#exponential 4.0
            # "vel_kd":    {"idx": 2, "default": 0.001,  "type": "linear", "scale": 1,    "clip": [0.00, 0.2] },#0.1
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

        # prepare reward functions and multiply reward scales by rl_dt (不是 dt!)
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.rl_dt  # [修改这里] RL 每步代表 0.02s 的时间累积
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

        # ========== [新增] 平滑指令与加速度限制 ==========
        self.smoothed_commands = torch.zeros_like(self.commands)
        # 物理加速度极限设为 2.5 m/s^2。这意味着加速到 2.0 m/s 需要 0.8 秒，非常平稳
        self.max_cmd_accel = 2.5

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
        if len(envs_idx) == 0:
            return

        num_reset = len(envs_idx)

        # --------------------------------------------------------
        # A. 采样 Linear Velocity X (四级疫苗式混合采样)
        # --------------------------------------------------------
        weights = torch.tensor(self.command_cfg.get("sampling_weights", [0.5, 0.3, 0.1, 0.1]), device=self.device)
        bucket_indices = torch.multinomial(weights, num_reset, replacement=True)
        
        new_vel_x = torch.zeros(num_reset, device=self.device)

        # 1. Bucket 0: 核心测试点 (50%) -> 取 0.5, 1.0, 1.5 的正负
        mask_0 = (bucket_indices == 0)
        num_0 = int(mask_0.sum().item())
        if num_0 > 0:
            test_points = torch.tensor(self.command_cfg["test_points"], device=self.device)
            pts_idx = torch.randint(0, len(test_points), (num_0,), device=self.device)
            base_v = test_points[pts_idx]
            signs = torch.randint(0, 2, (num_0,), device=self.device) * 2 - 1
            # 加一点微小噪声防死记硬背
            noise = gs_rand_float(-0.05, 0.05, (num_0,), self.device)
            new_vel_x[mask_0] = base_v * signs + noise

        # 2. Bucket 1: 连续主训练区 (30%) -> [0.5, 2.0] 和 [-2.0, -0.5]
        mask_1 = (bucket_indices == 1)
        num_1 = int(mask_1.sum().item())
        if num_1 > 0:
            v_mag = gs_rand_float(0.5, 2.0, (num_1,), self.device) 
            signs = torch.randint(0, 2, (num_1,), device=self.device) * 2 - 1
            new_vel_x[mask_1] = v_mag * signs

        # 3. Bucket 2: 绝对静止 (10%) -> 0.0
        mask_2 = (bucket_indices == 2)
        num_2 = int(mask_2.sum().item())
        if num_2 > 0:
            new_vel_x[mask_2] = gs_rand_float(-0.01, 0.01, (num_2,), self.device)

        # 4. Bucket 3: 疫苗低速区 (10%) -> [0.02, 0.5] 和 [-0.5, -0.02]
        mask_3 = (bucket_indices == 3)
        num_3 = int(mask_3.sum().item())
        if num_3 > 0:
            v_mag = gs_rand_float(0.02, 0.5, (num_3,), self.device)
            signs = torch.randint(0, 2, (num_3,), device=self.device) * 2 - 1
            new_vel_x[mask_3] = v_mag * signs

        # 最终赋值并用硬限幅保护在 [-2.0, 2.0]
        self.commands[envs_idx, 0] = torch.clamp(new_vel_x, -2.0, 2.0)

        # --------------------------------------------------------
        # B. 采样 Angular Velocity (角速度) - [完美复刻你的物理约束]
        # --------------------------------------------------------
        if self.command_cfg.get("limit_cmd_random", False):
            R = self.command_cfg["wheel_radius"]
            max_w = self.command_cfg["wheel_max_w"]
            spacing = self.command_cfg["wheel_spacing"]
            
            v_lin = torch.abs(self.commands[envs_idx, 0])
            wheel_v_limit = max_w * R
            allowed_w = (wheel_v_limit - v_lin).clamp(min=0.0) * 2.0 / spacing
            
            raw_low = self.command_ranges[envs_idx, 2, 0]
            raw_high = self.command_ranges[envs_idx, 2, 1]
            
            final_low = torch.max(raw_low, -allowed_w)
            final_high = torch.min(raw_high, allowed_w)
            
            self.commands[envs_idx, 2] = gs_rand_float(final_low, final_high, (len(envs_idx),), self.device)
            self.commands[envs_idx, 1] = gs_rand_float(
                self.command_ranges[envs_idx, 1, 0],
                self.command_ranges[envs_idx, 1, 1], 
                (len(envs_idx),), self.device
            )
            if self.command_ranges.shape[1] > 3:
                self.commands[envs_idx, 3] = gs_rand_float(
                    self.command_ranges[envs_idx, 3, 0], 
                    self.command_ranges[envs_idx, 3, 1], 
                    (len(envs_idx),), self.device
                )
            
        else:
            for cmd_i in range(self.num_commands):
                self.commands[envs_idx, cmd_i] = gs_rand_float(
                    self.command_ranges[envs_idx, cmd_i, 0],
                    self.command_ranges[envs_idx, cmd_i, 1],
                    (len(envs_idx),), self.device
                )

        # 最后强制 Y 轴清零
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
        """专门负责向 TensorBoard 写入数据，反映 RL 对物理姿态的真实干预"""
        if self.writer is None:
            return

        env_id = 0 # 默认记录第 0 个环境的数据

        # 1. 记录 RL 输出的残差偏置 (Residual Actions)
        # 这是重构后的核心：观察大脑对小脑的干预强度
        lqr_params = self.lqr_controller.params
        vmc_params = self.vmc_controller.params

        if 'rl_pitch_offset' in lqr_params:
            # 记录俯仰补偿（弧度转度，更直观）
            pitch_offset_deg = lqr_params['rl_pitch_offset'][env_id] * 180 / math.pi
            self.writer.add_scalar('RL_Actions/Pitch_Offset_Deg', pitch_offset_deg, self.log_step)
        
        if 'rl_delta_z_left' in vmc_params:
            # 记录左右腿独立高度偏置 (单位：米)
            self.writer.add_scalar('RL_Actions/Delta_Z_Left', vmc_params['rl_delta_z_left'][env_id], self.log_step)
            self.writer.add_scalar('RL_Actions/Delta_Z_Right', vmc_params['rl_delta_z_right'][env_id], self.log_step)

        # 2. 记录基础运动跟踪 (Tracking)
        # 帮助分析“偏航”问题是否得到改善
        vel_error = self.commands[env_id, 0] - self.base_lin_vel[env_id, 0]
        theta_pitch = self.lqr_controller.theta_pitch[env_id]

        self.writer.add_scalar('Tracking/Vel_X_Error', vel_error, self.log_step)
        self.writer.add_scalar('Tracking/Command_Vel', self.commands[env_id, 0], self.log_step)
        self.writer.add_scalar('Tracking/Real_Vel_X', self.base_lin_vel[env_id, 0], self.log_step)
        self.writer.add_scalar('Tracking/Theta_Pitch', theta_pitch, self.log_step)
        # 记录基座离地高度（反映避障时的整体高度控制）
        self.writer.add_scalar('Tracking/Base_Height', self.base_pos[env_id, 2], self.log_step)

        # 3. 记录机体姿态 (Attitude)
        r = self.base_euler[env_id, 0]
        p = self.base_euler[env_id, 1]
        y = self.base_euler[env_id, 2]
        p_vel = self.base_ang_vel[env_id, 1]

        gamma_d_g = self.lqr_controller.params['gamma_d_g'][env_id] if 'gamma_d_g' in self.lqr_controller.params else 0.0
        eth = self.lqr_controller.params['eth'][env_id] if 'eth' in self.lqr_controller.params else 0.0
        a = self.lqr_controller.params['a'][env_id] if 'a' in self.lqr_controller.params else 0.0


        

        self.writer.add_scalar('Params/Gamma_D_G', gamma_d_g, self.log_step)
        self.writer.add_scalar('Params/Eth', eth, self.log_step)
        self.writer.add_scalar('Params/A', a, self.log_step)

        self.writer.add_scalar('Attitude/Roll', r , self.log_step)
        self.writer.add_scalar('Attitude/Pitch', p , self.log_step)
        self.writer.add_scalar('Attitude/Yaw', y , self.log_step)
        self.writer.add_scalar('Attitude/Pitch_Vel', p_vel, self.log_step)

        # 4. 记录控制器内部状态 (Internal Controller State)
        # p_com 是 LQR 根据当前腿长算出的静态平衡俯仰角
        p_com = self.lqr_controller.pitch_com[env_id]
        self.writer.add_scalar('Controller/Pitch_CoM_Reference', p_com, self.log_step)

        # 5. 记录奖励总和 (Reward)
        # 方便观察本步奖励的波动
        self.writer.add_scalar('Reward/Step_Total', self.rew_buf[env_id], self.log_step)

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
        
        # [新增修复] 在 last_actions 被覆盖前，先计算出差值，专门留给 Reward 用
        self.reward_action_diff = self.actions - self.last_actions

        
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
        
# ================== [新增核心] 重置缓冲与指令斜坡滤波 ==================
        
        # 1. 缓冲期判定：存活时间小于 0.5 秒 (0.5 / rl_dt) 时，强制目标指令为 0
        # 假设 rl_dt = 0.02s，则前 25 个 step 机器人只会原地站立
        warm_up_steps = int(0.5 / self.rl_dt)
        is_warming_up = self.episode_length_buf < warm_up_steps
        
        # 2. 提取当前真实目标指令 (如果是缓冲期，目标强制覆写为 0)
        active_target_commands = self.commands.clone()
        active_target_commands[is_warming_up, :] = 0.0
        
        # 3. 计算单步允许的最大速度变化量 (delta_v = a * dt)
        max_delta_v = self.max_cmd_accel * self.rl_dt
        
        # 4. 计算指令差值，并用 clamp 限制在最大加速度范围内
        cmd_diff = active_target_commands - self.smoothed_commands
        cmd_diff = torch.clamp(cmd_diff, -max_delta_v, max_delta_v)
        
        # 5. 更新平滑指令
        self.smoothed_commands += cmd_diff

        # ===================================================================

        self.lqr_controller.set_commands(
            velocity_d = self.smoothed_commands[:, 0], # [关键] 这里必须传入平滑后的指令！
            yaw_d = self.commands[:, 2] * 0,  # 同样传入平滑后的 yaw
        )
        self.vmc_controller.set_commands(
            height_target = self.height_target 
        )

        # ==========================================================
        # 2. 【控制与物理层】 (500Hz)：内部高频循环
        # ==========================================================
        for _ in range(self.decimation):
            # 获取当前物理真值 (必须在循环内不断获取最新状态！)

            # [核心修改] 传感器模拟层：提前加噪，然后喂给控制器
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

            leg_torques = self.vmc_controller.vmc()      # [num_envs, 4]
            wheel_torques = self.lqr_controller.balance()  # [num_envs, 2]        
            # print(f"leg_torques: {leg_torques}")
            # print(f"wheel_torques: {wheel_torques}")

            # ============= 传递力矩给机器人 =============
            self.robot.control_dofs_force(leg_torques, self.motor_dofs[0:4])
            self.robot.control_dofs_force(wheel_torques, self.motor_dofs[4:6])

# 🚀 【核心修复 1】：调用纯物理底层，彻底抛开渲染的拖累！
            # 替换掉原来的 self.scene.step()
            self.scene.sim.step() 

        # 🚀 【核心修复 2】：退出 10 次物理循环后 (即过去 0.02s)，手动刷新一次画面
        # 相当于你 MuJoCo 代码里的 viewer.sync()
        if hasattr(self.scene, '_visualizer') and self.scene._visualizer is not None:
            self.scene._visualizer.update()
        # ==========================================================
        # 3. 【仿真状态更新与观测生成】 (50Hz)
        # ==========================================================
        # 更新 Buffer (仅保留循环结束后的最终状态供 RL 使用)
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_theta_pitch = self.lqr_controller.theta_pitch.clone()  # 存储上一步的 theta_pitch，供奖励函数使用
        # 注意：self.last_actions[:] = self.actions[:] 已经在循环外第一步处理过了

        self.episode_length_buf += 1  # 此时步数 +1，代表经过了 0.02s

        self.base_pos[:] = self.get_relative_terrain_pos(self.robot.get_pos())
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_lin_acc[:] = (self.base_lin_vel[:] - self.last_base_lin_vel[:])/ self.rl_dt
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat) 
        self.base_ang_acc[:] = (self.base_ang_vel[:] - self.last_base_ang_vel[:]) / self.rl_dt
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat) 

        # =======================================================================

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs) 
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs) 
        self.dof_force[:] = self.robot.get_dofs_force(self.motor_dofs)
        self.theta_pitch = self.lqr_controller.theta_pitch



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

        
        #步数
        self.episode_lengths += 1


        # resample commands
        # envs_idx = (
        #     (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.rl_dt) == 0)
        #     .nonzero(as_tuple=False)
        #     .flatten()
        # )

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

        # if(self.mode):
        #     self._resample_commands(envs_idx)
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

        self.slice_obs_buf = torch.cat([
            self.base_lin_vel ,   # 脏
            self.base_ang_vel ,   # 脏
            self.projected_gravity,
            self.commands ,
            (self.dof_pos[:,0:4] - self.default_dof_pos[:,0:4]), # 脏
            self.dof_vel , #到此23维       # 脏
            self.actions ,
            # all_phy_params , 
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
            # all_phy_params,
            clean_pitch_obs                                   # 净
        ], axis=-1)




        
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

        # ==================== 1. [核心修改] 先分配新指令 ====================
        # 必须先清空平滑指令并采样新指令，后面才能根据指令正负安排位置
        self.smoothed_commands[envs_idx] = 0.0
        self._resample_commands(envs_idx)
        
        # 如果开启了域随机化，同步进行
        if self.mode:
            self.domain_rand(envs_idx)

        # ==================== 2. 重置关节与状态 ====================
        self.dof_pos[envs_idx] = self.init_dof_pos[envs_idx]  
        self.dof_vel[envs_idx] = 0.0  
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)  

        # ==================== 3. [核心修改] 根据指令方向分配出生点 ====================
        if self.terrain_cfg["terrain"]:  
            # 获取刚刚采样出的目标线速度
            target_vel_x = self.commands[envs_idx, 0]
            
            # 生成布尔掩码：True 表示前进 (应该出生在索引 0)，False 表示倒车 (出生在索引 1)
            is_forward = target_vel_x >= 0
            
            # 生成位置索引张量：is_forward 为 True 填 0，否则填 1
            spawn_indices = torch.where(
                is_forward,
                torch.zeros_like(envs_idx, dtype=torch.long),
                torch.ones_like(envs_idx, dtype=torch.long)
            )
            
            # 批量分配坐标
            self.base_pos[envs_idx] = self.base_terrain_pos[spawn_indices]
        else:
            self.base_pos[envs_idx] = self.base_init_pos
            
        # 设置到底层物理引擎
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # ==================== 4. 重置控制器与历史记忆 ====================
        self.lqr_controller.command_pitch.prev_error[envs_idx] = 0.0
        self.lqr_controller.command_pitch.integral[envs_idx] = 0.0
        self.lqr_controller.command_velocity.prev_error[envs_idx] = 0.0
        self.lqr_controller.command_velocity.integral[envs_idx] = 0.0
        self.lqr_controller.command_yaw.prev_error[envs_idx] = 0.0
        self.lqr_controller.command_yaw.integral[envs_idx] = 0.0

        self.obs_history_buf[envs_idx] = 0.0
        self.clean_history_buf[envs_idx] = 0.0
        self.lqr_controller.reset_params(envs_idx)

        self.actions[envs_idx] = 0.0      
        self.last_actions[envs_idx] = 0.0 
        self.last_dof_vel[envs_idx] = 0.0
        
        self.reset_buf[envs_idx] = True
        self.episode_length_buf[envs_idx] = 0 
        self.episode_lengths[envs_idx] = 0.0
        
        # ⚠️ 注意：函数末尾无需再调用 self._resample_commands(envs_idx)，因为第一步已经做过了





    def reset(self):
        self.reset_buf[:] = True  # 用于标志环境是否需要reset（全部）
        self.reset_idx(torch.arange(self.num_envs, device=self.device))  # 根据索引去reset环境
        return self.obs_buf, None

    def check_termination(self):
        # 1. 超时与姿态崩溃判定 (保留原有)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        
        # 2. [核心修改] 双向冲线判定
        # 前进冲线：目标速度 > 0 且 机器人 X 坐标超出了 finish_line_x_end
        forward_finish = (self.commands[:, 0] >= 0) & (self.base_pos[:, 0] > self.finish_line_x_end)
        
        # 倒车冲线：目标速度 < 0 且 机器人 X 坐标低于了 finish_line_x_start
        backward_finish = (self.commands[:, 0] < 0) & (self.base_pos[:, 0] < self.finish_line_x_start)
        
        # 满足任意一个即可重置，并开启下一轮测试
        self.reset_buf |= (forward_finish | backward_finish)
        
        # 3. 触地判定 (保留原有)
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
            if self.episode_lengths[envs_idx].mean()/(self.env_cfg["episode_length_s"]*self.stiffness_end/self.rl_dt) > self.damping_threshold:
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
                                                                          *self.stiffness_end/self.rl_dt))
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
    # [兼顾测试指标与底层物理平滑的终极奖励函数]
    # =======================================================================================================================
    
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.smoothed_commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_lin_sigma"])

    def _reward_inner_pitch_error(self):
        # 【修改】使用指数映射 (Bonus 机制)。
        # pitch_ref = 目标俯仰角, pitch_real = 实际俯仰角
        error = self.lqr_controller.theta_pitch
        error_sq = torch.square(error)
        # 误差为0时，获得最大正分；误差极大时，得分为0(但不扣分，允许加速)

        return torch.exp(-error_sq / self.reward_cfg["inner_pitch_sigma"])
    
    # 新增：收敛速度奖励（需要在控制器里保存last_theta_pitch）
    def _reward_inner_pitch_convergence(self):
        error_now  = torch.abs(self.theta_pitch)
        error_last = torch.abs(self.last_theta_pitch)  # 需新增
        delta = error_last - error_now   # 正值=误差在减小=好
        is_large_error = error_now > 0.10  # 只在误差较大时激活
        return torch.clamp(delta, min=0.0) * is_large_error.float()

    def _reward_tracking_roll(self):
        # return torch.square(self.base_euler[:, 0])
        return torch.square(self.lqr_controller.roll)


    def _reward_pitch_vel(self):
        return torch.square(self.lqr_controller.pitch_dot)

    # ================= [新增] 物理底层平滑度约束 =================
    def _reward_action_rate(self):
        # 【修复1】使用正确保存的动作差值，直接返回平方和 (无 exp)
        sq_diff = torch.sum(torch.square(self.reward_action_diff), dim=1) 
        return sq_diff

    def _reward_dof_acc(self):
        # 【修复2】除以 rl_dt (0.02s)，计算正确的宏观加速度
        dof_acc = (self.dof_vel - self.last_dof_vel) / self.rl_dt
        return torch.sum(torch.square(dof_acc), dim=1)

    def _reward_base_accel_x(self):
        # 【修复2】除以 rl_dt
        acc = (self.base_lin_vel[:, 0] - self.last_base_lin_vel[:, 0]) / self.rl_dt
        return torch.square(acc)

    def _reward_knee_height(self):
        left_knee_low = self.left_knee_pos[:, 2] < 0.08
        right_knee_low = self.right_knee_pos[:, 2] < 0.08
        return (left_knee_low.float() + right_knee_low.float())