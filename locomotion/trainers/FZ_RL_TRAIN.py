import argparse
import os
import pickle
import shutil

from environments.FZ_RL_ENV import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs # type: ignore


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.18,
            "desired_kl": 0.005,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 3,
            "num_mini_batches": 5,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            # [修改 1] Actor 不需要那么深，去掉最后的 64，[512, 256, 128] 是标准配置
            # 这里的 512 能保证有足够的特征提取能力处理观测历史
            "actor_hidden_dims": [512, 256, 128],
            
            # [修改 2] Critic 必须加强！让它和 Actor 保持一致
            # 只有 Critic 足够强，才能从复杂的特权信息中提取出正确的"上帝指引"
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.5,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 30,    #每轮仿真多少step
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,  # 启用断点续传
            "resume_path": "locomotion/logs/wl_10000/model_10000.pt",  # 指定要加载的模型路径
            "run_name": "wl_15000",
            "runner_class_name": "runner_class_name",
            "save_interval": 25,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 3,
        "num_dofs": 6,
        # joint names
        "default_joint_angles": {  # [rad]
            "L1_joint": -0.2800,#-0.28
            "L2_joint": 0.3385,#0.338
            "R1_joint": 0.2800,#0.28
            "R2_joint": -0.3385,#-0.338
            "L3_joint": 0.0,
            "R3_joint": 0.0,
        },
        "joint_init_angles": {  # [rad]
            "L1_joint": -0.2800,#-0.28
            "L2_joint": 0.3385,#0.338
            "R1_joint": 0.2800,#0.28
            "R2_joint": -0.3385,#-0.338
            "L3_joint": 0.0,
            "R3_joint": 0.0,
        },
        "dof_names": [
            "L1_joint",  # 大腿
            "L2_joint",    # 小腿
            "R1_joint",
            "R2_joint",
            "L3_joint",   # 轮子
            "R3_joint",
        ],
        # lower upper
        # 物理引擎的配置优先于urdf的配置
        # 由于wheel的joint设置是continuous，而其他关节是revolution。所以肯wheel的关节没有角度的限制，但是需要有速度限制和扭矩限制
        "dof_limit": {
            "L1_joint": [-0.39, 1.05],  # 左大腿关节：垂直时为1.2217弧度(70°)，前后可摆动70°(1.2217弧度)
            "L2_joint": [-1.40, 0.75],  # 左小腿关节：大小腿一条直线时为-1.663弧度(-95.3°)，逆时针可旋转100°(1.7453弧度)至-0.082弧度(4.7°)
            "R1_joint": [-1.05, 0.39],  #轴为1时候：-2.82, 1.95
            "R2_joint": [-0.75, 1.40],  #轴为1时候：-0.89, 3.14
            "L3_joint": [0.0, 0.0],  # 固定
            "R3_joint": [0.0, 0.0],  # 疑惑！
        },
        "safe_force": {
            "L1_joint": 20.0,   # 左大腿关节最大扭矩是20N·M，增大数值可以输出更大的力，可以快速运动，但是可能损坏电机或者结构
            "L2_joint": 20.0,
            "R1_joint": 20.0,
            "R2_joint": 20.0,
            "L3_joint": 100.0, #给大点，测试的时候可以小
            "R3_joint": 100.0,
        },
        # PD
        # 先增大 Kp，直到系统出现轻微震荡。再增大 Kv，使震荡平息。

        "joint_kp": 0,  # 对位置误差进行放大。较大的 Kp 能让系统响应更迅速，不过可能引发震荡。若 Kp过小，关节动作会变得迟缓；若过大，则可能导致关节剧烈抖动
        "joint_kv": 1.5,  # 对误差变化率起到阻尼作用，可抑制震荡，若 Kv​ 过小，系统可能会持续震荡；若过大，系统响应会变得缓慢
        "wheel_kp": 0,
        "wheel_kv": 15,
        "damping": 0.1,  # 模拟关节的黏性阻力，其效果和 Kv类似，但它是物理层面的属性，较小的阻尼系数会使关节运动更灵活
        "stiffness":0.0, #不包含轮。  # 关节刚度。定义关节的弹性特性，值为 0 表示关节是刚性的
        "armature":0.004,  # 电机惯性。模拟电机转子的惯性，数值越小，电机响应越迅速。需要与电机参数匹配
        # termination 角度制    obs的angv弧度制
        # 终止条件的参数
        "termination_if_roll_greater_than": 35,  # degree  当机器人基座的侧倾角超过 20 度时，回合终止
        "termination_if_pitch_greater_than": 35, #15度以内都摆烂，会导致episode太短难以学习
        # "termination_if_base_height_greater_than": 0.1,
        # "termination_if_knee_height_greater_than": 0.00,
        "termination_if_base_connect_plane_than": True, #触地重置
        "connect_plane_links":[ #触地重置link
            "base_link",
            "L1_link",
            "L2_link",
            "R1_link",
            "R2_link",
                ],
        # base pose
        "base_init_pos":{
            "urdf":[0.0, 0.0, 0.38],#稍微高一点点
            "mjcf":[0.0, 0.0, 0.285],
            },
        # 机器人基座的初始姿态
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],#0.996195, 0, 0.0871557, 0， 这里是完全数值，没有旋转。初始姿态影响学习难度（如倾斜姿态增加平衡挑战）。sim2real需要与仿真有一样的初始姿态
        "episode_length_s": 10.0,  # 每个训练回合的最大持续时间（秒）。过短容易重启，过大容易无效探索
        "resampling_time_s": 1.5,
        # "joint_action_scale": 0.5,
        # "wheel_action_scale": 10,
        "simulate_action_latency": True,
        # "clip_actions": 100.0,
    }
    obs_cfg = {
        # num_obs = num_slice_obs + history_num * num_slice_obs
        "num_obs": 31*6, #在rsl-rl中使用的变量为num_obs表示state数量
        "num_slice_obs": 31,
        "history_length": 5,
        "obs_scales": {
            "lin_vel": 1,
            "ang_vel": 1,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
        },
        "noise":{
            "use": True,
            #[高斯,随机游走]
            "ang_vel": [0.001,0.0011],
            "dof_pos": [0.0001,0.0005],
            "dof_vel": [0.0001,0.0005],
            "gravity": [0.0001,0.002],
            "base_euler": [0.0001,0.0005],
        }
    }
    # 名字和奖励函数名一一对应
# 名字和奖励函数名一一对应
    reward_cfg = {
# Sigma 定义了“多少误差算差”
    "tracking_lin_sigma": 0.05, 
    "tracking_ang_sigma": 0.05, 
    "tracking_pitch_sigma": 2,  # 稍微放松，允许加速时的俯仰
    "tracking_roll_sigma": 2,
    "action_rate_sigma": 0.05,     # 收紧！强迫丝滑
    "action_magnitude_sigma": 0.5, # 适度宽松
    
        "reward_scales": {
            # =========== 正向奖励 (所有项必须为正) ===========
            
            # 1. 核心任务 (权重最高)
            "tracking_lin_vel": 4.0,     # 主任务，一定要最大
            "tracking_ang_vel": 0.3,     # 辅助任务
            
            # 2. 姿态稳定 (安全基石)
            "tracking_pitch": 1,      # 保持姿态分
            "tracking_roll": 1,       # 保持姿态分
            # 'roll': 0.0,                # 直接限制roll角，防止侧翻
            # 3. 动作约束 (正则化)
            # 解释：这是“平滑奖励”。越平滑，得分越高(max 0.1)。
            # 不平滑只是拿不到这 0.1 分，而不会倒扣分。
            # 这种 "Bonus" 机制比 "Penalty" 机制训练起来更温和。
            "action_rate": 0.5,          
            "action_magnitude": 0.1,     
            
            # 4. 必要的惩罚 (依然建议保留传统的负分项)
            # 对于绝对不能发生的事情（如膝盖撞地、飞车），传统的负无穷惩罚更有效
            "knee_height": -1.0,         # 绝对禁止跪地
            "collision": -1.0,           # 绝对禁止乱撞

            # =========== 2. 机械健康 (你强调的) ===========
            # 引导机器人保持标准站姿，权重适中，作为长期引导
            "joint_pose_health": 0.0,     
            # 坚决防止跪地
      

            # =========== 3. 姿态红线 ===========
            # 超过 20 度才扣分。
            # 系数 -0.5 意味着：25度时扣 12.5分(抵消速度分)，30度时扣 50分(不可接受)
            "lqr_pitch_position": -0.000,   
            # "roll": -0e-11,
            
            # 抑制震荡，保证看起来很稳
            "lqr_pitch_dot": -0e-4,#-1.5
            "pitch_vel": -0e-5,#-1.5

            "ang_vel_xy": -0.0,
            "lin_vel_z": -0.0,
            "ang_vel_xy_dot": -0.0,

            # =========== 4. 平滑与能耗 ===========
            # 这里的惩罚迫使 RL 即使在追速度时，也不能把 PID 调得太激进
            "base_accel_x": -0,         # 允许适度推背感，但不许顿挫
            # "dof_acc": -2.0e-7,           # 保护关节
            # "dof_force": -2.0e-8,         # 节能，优化 PID 效率
            # "dof_vel": -1.0e-8,           # 防止动作过大
            "dof_acc": -1.0e-6,           # 保护关节
            "dof_force": 0,         # 节能，优化 PID 效率
            "dof_vel": -0,           # 防止动作过大
            # =========== 5. 参数约束 ===========
            # 9维动作下，我们希望参数在默认值附近微调

            # [新增] 停滞惩罚权重
            # 这个权重不需要太大，大概 -1.0 到 -2.0 左右
            # 只要能打破“躺平收益 > 0”的平衡即可
            "stall": -0,

            # [新增] 软超调惩罚
            # 权重建议给重一点 (-5.0 到 -10.0)
            # 因为我们已经给了它 10% 的“免死金牌”了，如果它还冲出 10% 之外，说明失控了，必须重罚。
            "vel_overshoot_soft": -0e-5,
        },
    }

    command_cfg = {
            "num_commands": 4,
            "base_range": 1,
            # 基础物理范围 (你可以适当开大一点，测试 RL 的极限)
            "lin_vel_x_range": [-2, 2], 
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [-3.14, 3.14],
            "height_target_range": [0.22, 0.40],
            
            # [重要] 开启物理约束，防止生成轮子转不过来的超速指令
            "limit_cmd_random": True, 
            
            "wheel_spacing": 0.37,
            "wheel_radius": 0.075,   # 0.075? 请确认单位是米。如果是 0.75米这轮子太大了
            "wheel_max_w": 20,      # 提高一点最大转速限制，给 RL 更多空间
            
            # [核心] 概率分桶策略
            # 既然 PID 基准很好，我们要重点练 "高速" 和 "急停"
            "distribution_buckets": [
                # 1. 后退区 (30%)
                {"range_pct": [0.0, 0.5], "prob": 0.5},
                
                # 2. 前进低速区 (20%) - 简单巡航，送分题
                {"range_pct": [0.5, 0.8], "prob": 0.2},
                
                # 3. 前进高速区 (50%) - 重点突破！
                # 对应范围: 0.9m/s ~ 1.5m/s
                {"range_pct": [0.8, 1.0], "prob": 0.3},
            ]
        }
    # 课程学习，奖励循序渐进 待优化
    curriculum_cfg = {
        "curriculum_lin_vel_step":0.0001,   #比例
        "curriculum_ang_vel_step":0.0001,   #比例
        "curriculum_height_target_step":0.00003,   #高度
        "curriculum_lin_vel_min_range":0.3,   #比例
        "curriculum_ang_vel_min_range":0.3,   #比例
        "lin_vel_err_range":[0.05,0.5],  #课程误差阈值
        "ang_vel_err_range":[0.20,0.45],  #课程误差阈值 连续曲线>方波>不波动
        "damping_descent":True,
        "dof_damping_descent":[0.2, 0.05, 0.001, 0.4],#[damping_max,damping_min,damping_step（比例）,damping_threshold（存活步数比例）]
    }
    #域随机化 friction_ratio是范围波动 mass和com是偏移波动
    domain_rand_cfg = { 
        "friction_ratio_range":[0.70 , 1.6],
        "random_base_mass_shift_range":[-0.5 , 1], #质量偏移量
        "random_other_mass_shift_range":[-0.05, 0.05],  #质量偏移量
        "random_base_com_shift":0.05, #位置偏移量 xyz
        "random_other_com_shift":0.01, #位置偏移量 xyz
        "random_KP":[0.8, 1.2], #比例
        "random_KV":[0.8, 1.2], #比例
        "random_default_joint_angles":[-0,-0], #rad
        "damping_range":[0.9, 1.1], #比例
        "dof_stiffness_range":[0.0 , 0.0], #范围 不包含轮 [0.0 , 0.0]就是关闭，关闭的时候把初始值也调0
        "dof_stiffness_descent":[0.0 , 0.0], #刚度下降[max_stiffness，仿真步数占比阈值]，和域随机化冲突，二选一
        "dof_armature_range":[0.0 , 0.008], #范围 额外惯性 类似电机减速器惯性 有助于仿真稳定性

        # [新增] 重心高度变化 (模拟背了东西)
        "random_base_com_shift": 0.05, 
        
        # 延迟随机化 (Sim2Real 神器)
        "random_lag_timesteps": [0, 2], # 模拟 0ms ~ 20ms 通信延迟
    }
    #地形配置
    terrain_cfg = {
        "terrain":True, #是否开启地形
        "train":"agent_train_gym",
        "eval":"agent_eval_gym",    # agent_eval_gym/circular
        "num_respawn_points":3,
        "respawn_points":[
            [-5.0, -5.0, 0.0],    #plane地形坐标，一定要有，为了远离其他地形
            [5.0, 5.0, 0.0],
            [15.0, 5.0, 0.08],
        ],
        "horizontal_scale":0.1,
        "vertical_scale":0.001,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg
# [新增] 必须导入这个，防止 runner.writer 是空的
from torch.utils.tensorboard import SummaryWriter
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="new_ts")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)



    args = parser.parse_args()

    gs.init(logging_level="warning",backend=gs.gpu)
    gs.device="cuda:0"
    log_dir = f"locomotion/logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)



    # 断点续传时不删除现有日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    env = WheelLeggedEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, 
        command_cfg=command_cfg, curriculum_cfg=curriculum_cfg, 
        domain_rand_cfg=domain_rand_cfg, terrain_cfg=terrain_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

# 2. [优雅挂载] 解决 Runner 懒加载导致的 None 问题
    # 我们直接手动创建一个 Writer 指向 log_dir，确保万无一失
    # TensorBoard 会自动合并同一目录下的数据

    print(f"🔗 正在挂载训练日志到: {log_dir}")
    
    # 调用 Env 的标准接口
    env.set_logger(SummaryWriter(log_dir=log_dir))



#############
    # 手动加载模型进行断点续传
    if train_cfg["runner"]["resume"] and train_cfg["runner"]["resume_path"]:
        resume_path = train_cfg["runner"]["resume_path"]
        if os.path.exists(resume_path):
            print(f"正在加载模型: {resume_path}")
            infos = runner.load(resume_path, load_optimizer=True)
            print(f"成功加载模型，当前迭代次数: {runner.current_learning_iteration}")
            if infos:
                print(f"模型信息: {infos}")
        else:
            print(f"警告: 指定的模型文件不存在: {resume_path}")
            print("将从头开始训练")
#############
    """
    实验复现：确保后续训练使用相同配置
    断点续训：恢复中断的训练
    配置共享：在不同设备或实验间传递参数
    """
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),  # 以二进制写入模式打开文件
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

