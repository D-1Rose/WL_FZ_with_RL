import copy
# 使用新的导入路径
try:
    from locomotion.controllers.FZ_PID_c import BatchedLQRController
    from locomotion.controllers.FZ_PID_c import BatchedVMC
except ImportError:
    # 回退到旧路径（通过文件索引系统重定向）
    from FZ_PID_c import BatchedLQRController
    from FZ_PID_c import BatchedVMC
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
        self.action_smooth_factor = 0.60  # 平滑系数：0.65足够平滑但保持快速响应（不要太大）
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

        # ================== [移植] 平滑指令与加速度限制 ==================
        self.max_cmd_accel = 1.5  # 物理加速度极限设为 2.5 m/s^2

        # ================== [移植] 双向冲线坐标 ==================
        self.finish_line_x_start = -2  # 倒车冲线坐标
        self.finish_line_x_end = 8     # 前进冲线坐标
        # self.max_episode_length = 650   # 兜底超时

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

        


        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                # max_FPS=60,
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
                        file="assets/description/u/wheel_leg.urdf",
                        pos = base_init_pos,
                        quat=self.base_init_quat.cpu().numpy(),
                    ),
                )
            # case "mjcf":
            #     self.robot = self.scene.add_entity(
            #         gs.morphs.MJCF(file="assets/mjcf/nz/nz_view.xml",
            #         pos=base_init_pos),
            #         vis_mode='collision'
            #     )
            case _:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file="assets/description/u/wheel_leg.urdf",
                        pos = base_init_pos,
                        quat=self.base_init_quat.cpu().numpy(),
                    ),
                )
        # build
        self.scene.build(n_envs=num_envs)



        self.lqr_controller = BatchedLQRController(num_envs=self.num_envs,
                                                   device=self.device)
        self.vmc_controller = BatchedVMC(num_envs=self.num_envs, 
                                         device=self.device)

        # ================== 参数注册表 (The Registry) ==================
        # 定义动作如何映射到参数，并增加 "clip" 字段进行安全截断
        self.param_registry = {
        }



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
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["height_measurements"]], 
            device=self.device,
            dtype=gs.tc_float,
        )

        self.smoothed_commands = torch.zeros_like(self.commands)

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
        lqr_params = self.lqr_controller.params
        self.writer.add_scalar('RL_Action/gamma_d_g', lqr_params['gamma_d_g'][env_id].item(), self.log_step)
        self.writer.add_scalar('RL_Action/eth', lqr_params['eth'][env_id].item(), self.log_step)
        self.writer.add_scalar('RL_Action/a', lqr_params['a'][env_id].item(), self.log_step)

        self.writer.add_scalar('Reward/reward', self.rew_buf[env_id].item(), self.log_step)

        # [完善] 记录 3维动作 (原汁原味，不带角度转换)


        # 2. 记录基础运动跟踪 (Tracking)
        vel_error = self.commands[env_id, 0] - self.base_lin_vel[env_id, 0]
        self.writer.add_scalar('Tracking/Vel_X_Error', vel_error, self.log_step)
        self.writer.add_scalar('Tracking/Command_Vel', self.commands[env_id, 0], self.log_step)
        self.writer.add_scalar('Tracking/Real_Vel_X', self.base_lin_vel[env_id, 0], self.log_step)
        self.writer.add_scalar('Tracking/Base_Height', self.base_pos[env_id, 2], self.log_step)
        
        # [完善] 记录 Z 轴线速度 (核心滤震指标，对应 _reward_base_z_vel)
        self.writer.add_scalar('Tracking/Real_Vel_Z', self.base_lin_vel[env_id, 2], self.log_step)

        # 3. 记录机体姿态 (Attitude) - [完善] 去除角度转换，纯弧度记录
        r = self.base_euler[env_id, 0]
        p = self.base_euler[env_id, 1]
        y = self.base_euler[env_id, 2]
        r_vel = self.base_ang_vel[env_id, 0] # 侧倾角速度
        p_vel = self.base_ang_vel[env_id, 1] # 俯仰角速度
        
        self.writer.add_scalar('Attitude/Roll', r, self.log_step)
        self.writer.add_scalar('Attitude/Pitch', p, self.log_step)
        self.writer.add_scalar('Attitude/Yaw', y, self.log_step)
        self.writer.add_scalar('Attitude/Roll_Vel', r_vel, self.log_step)   # 对应 _reward_roll_vel
        self.writer.add_scalar('Attitude/Pitch_Vel', p_vel, self.log_step)  # 对应 _reward_pitch_vel

        # 4. 记录控制器内部状态 (Internal Controller State)
        p_com = self.lqr_controller.pitch_com[env_id]
        self.writer.add_scalar('Controller/Pitch_CoM_Reference', p_com, self.log_step)
        
        # [完善] 记录系统整体指令高度
        if hasattr(self, 'smoothed_commands'):
            self.writer.add_scalar('Controller/Command_Height', self.smoothed_commands[env_id, 3], self.log_step)

        # 5. 记录奖励总和 (Reward)
        self.writer.add_scalar('Reward/Step_Total', self.rew_buf[env_id], self.log_step)

        # 计数器自增
        self.log_step += 1
        


    def scale_param(self, action, default, min_val, max_val, factor=0.2):
            """
            action: 网络原始输出
            default: 专家默认值
            min_val/max_val: 物理限幅
            factor: 敏感度因子。0.2 表示网络输出 1.0 时，参数偏移量程的 20%
            """
            span = max_val - min_val
            # 最终值 = 默认值 + 动作偏移
            param = default + action * (span * factor)
            return torch.clamp(param, min_val, max_val)    
    
    def step(self, actions):
        # 1. 动作截断
        # 为了防止训练初期极个别的离谱爆点扰乱平滑器，建议在动作空间层面加一个宽容的软截断（比如正负3或5）
        # 即使不加也没事，因为咱们后面算完参数有绝对的 clamp 保护。
        raw_actions = torch.clip(actions, -5.0, 5.0)
        # 2. 动作平滑 (低通滤波，极其重要，防止导数冲击)
        self.actions = self.action_smooth_factor * self.last_actions + \
                       (1 - self.action_smooth_factor) * raw_actions
        
        # 计算动作变化率 (供 reward_action_rate 惩罚使用)
        self.reward_action_diff = self.actions - self.last_actions
        
        # 3. 延迟模拟机制
        if self.simulate_action_latency:
            real_actions = self.last_actions.clone()
            self.last_actions = self.actions.clone()
        else:
            real_actions = self.actions.clone()
            self.last_actions = self.actions.clone()

        #  gamma_d_g (75°, [45°, 89°])
        gamma_d_g_default = 75.0 * math.pi / 180.0
        self.lqr_controller.params['gamma_d_g'] = self.scale_param(
            real_actions[:, 0], gamma_d_g_default, 
            45.0 * math.pi / 180.0, 89.0 * math.pi / 180.0, factor=0.3
        )

        #  eth (0.06, [0.01, 0.1])
        self.lqr_controller.params['eth'] = self.scale_param(
            real_actions[:, 1], 0.05, 0.01, 0.1, factor=0.2
        )

        #  a (0.06, [0.01, 0.99])
        self.lqr_controller.params['a'] = self.scale_param(
            real_actions[:, 2], 0.05, 0.01, 0.99, factor=0.2
        )
        

        # ================== [核心 2] 缓冲与指令斜坡滤波 ==================
        # 缓冲期判定：前 0.5 秒强制目标指令为 0，让机器人站稳
        warm_up_steps = int(0.1 / self.rl_dt)
        is_warming_up = self.episode_length_buf < warm_up_steps
        
        active_target_commands = self.commands.clone()
        active_target_commands[is_warming_up, :] = 0.0
        
        # 限制最大物理加速度
        max_delta_v = self.max_cmd_accel * self.rl_dt
        cmd_diff = active_target_commands - self.smoothed_commands
        cmd_diff = torch.clamp(cmd_diff, -max_delta_v, max_delta_v)
        
        self.smoothed_commands += cmd_diff

        # ================== [核心 3] 下发给控制器 ==================
        self.lqr_controller.set_commands(
            velocity_d = self.smoothed_commands[:, 0], 
            yaw_d_dot = self.commands[:, 2],  # 注意：你的 LQR 里接收的是 yaw_d_dot
        )
        self.vmc_controller.set_commands(
            height_target = self.commands[:, 3] 
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


        # [修复] 必须清空历史记忆！防止机器人带着上一局跌倒的残影开始新的一局
        self.obs_history_buf[envs_idx] = 0.0
        self.clean_history_buf[envs_idx] = 0.0
        
        # 【核心修复】将该环境的 RL 偏置清零
        self.lqr_controller.params['gamma_d_g'][envs_idx] = 75.0 * math.pi / 180.0
        self.lqr_controller.params['eth'][envs_idx] = 0.05
        self.lqr_controller.params['a'][envs_idx] = 0.05

# 2. 【补充】强制刷新一次观测堆叠
        self.lqr_controller.update_current_params()
            
        
        # ====== 以下两行直接删除（因为 VMC 和 LQR 已经没有这个函数了） ======
        # self.lqr_controller.reset_params(envs_idx) # 删除
        # self.vmc_controller.reset_params(envs_idx) # 删除


        # reset buffers
        # [修改] 重置动作 Buffer
        self.actions[envs_idx] = 0.0      # 重置平滑器
        self.last_actions[envs_idx] = 0.0 # 重置延迟队列
        self.last_dof_vel[envs_idx] = 0.0
        self.reset_buf[envs_idx] = True
        # [这里] 现在才清空步数，逻辑就对了！
        self.episode_length_buf[envs_idx] = 0 
        self.episode_lengths[envs_idx] = 0.0

# [核心修复] 必须把指令斜坡也清零，保证重生后乖乖缓冲 0.5 秒！
        self.smoothed_commands[envs_idx] = 0.0

        self._resample_commands(envs_idx)  
        if self.mode:
            self.domain_rand(envs_idx)





    def reset(self):
        self.reset_buf[:] = True  # 用于标志环境是否需要reset（全部）
        self.reset_idx(torch.arange(self.num_envs, device=self.device))  # 根据索引去reset环境
        return self.obs_buf, None

    def check_termination(self):
        # 1. 超时与姿态崩溃判定
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        
        # 2. [核心移植] 双向冲线判定
        # 前进冲线：目标速度 > 0 且 机器人 X 坐标超出了 finish_line_x_end
        forward_finish = (self.commands[:, 0] >= 0) & (self.base_pos[:, 0] > self.finish_line_x_end)
        
        # 倒车冲线：目标速度 < 0 且 机器人 X 坐标低于了 finish_line_x_start
        backward_finish = (self.commands[:, 0] < 0) & (self.base_pos[:, 0] < self.finish_line_x_start)
        
        # 满足任意一个即可重置，并开启下一轮测试
        self.reset_buf |= (forward_finish | backward_finish)
        
        # 3. 触地判定 
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
    # [重新设计的奖励函数]
    # =======================================================================================================================
    

# =================================================================================
    # 残差 RL 专属核心 Reward (替换掉你原来那些复杂的 pitch_offset_flat 等函数)
    # =================================================================================

    def _reward_tracking_lin_vel(self):
        # [主线任务] 速度跟踪
        lin_vel_error = torch.sum(torch.square(self.smoothed_commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_lin_sigma"])

    def _reward_inner_pitch_error(self):
        # 【修改】使用指数映射 (Bonus 机制)。
        # pitch_ref = 目标俯仰角, pitch_real = 实际俯仰角
        error = torch.abs(self.lqr_controller.theta_pitch)
        # 误差为0时，获得最大正分；误差极大时，得分为0(但不扣分，允许加速)

        return torch.exp(-error / self.reward_cfg["inner_pitch_sigma"])
    
    def _reward_pitch(self):
        # 【修改】使用指数映射 (Bonus 机制)。
        # pitch_ref = 目标俯仰角, pitch_real = 实际俯仰角
        error = torch.abs(self.lqr_controller.pitch)
        # 误差为0时，获得最大正分；误差极大时，得分为0(但不扣分，允许加速)
        return torch.exp(-error / self.reward_cfg["pitch_sigma"])

    def _reward_action_rate(self):
        # [防抽搐核心] 严厉惩罚 RL 动作（前馈偏置）的剧烈跳变
        # 逼迫 RL 输出平滑的斜坡指令，而不是高频矩形波
        return torch.sum(torch.square(self.reward_action_diff), dim=1)

    def _reward_action_norm(self):
        # [残差专属核心] 惩罚 RL 长期输出非 0 的偏置！
        # 告诉 RL：底层 PID 已经很棒了，在平地上你给我老老实实输出 0，只有遇到波浪不得已时才输出偏移量。
        return torch.square(self.actions[:, 0])

    def _reward_dof_acc(self):
        # [电机保护] 惩罚电机高频震荡
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel)/self.rl_dt), dim=1)
        
    def _reward_pitch_vel(self):
        # [姿态稳定] 惩罚机体高频点头
        return torch.square(self.lqr_controller.pitch_dot)
    


    # ================= [新增] 核心：主动悬挂的灵魂 =================
    def _reward_base_z_vel(self):
        # 【严惩底盘Z轴震荡】
        # 如果波浪的起伏被腿部完美吸收，底盘在Z轴的绝对速度应该趋近于0
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_roll_vel(self):
        # 【严惩机体左右摇晃】
        # 配合左右腿的独立伸缩，逼迫网络学会抗侧倾
        return torch.square(self.base_ang_vel[:, 0])
    

