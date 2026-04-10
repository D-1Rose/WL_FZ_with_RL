import argparse
import os
import pickle
import shutil

from wheel_legged_env import WheelLeggedEnv
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
            "actor_hidden_dims": [512, 256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
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
            "save_interval": 250,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,
        # joint names
        "default_joint_angles": {  # [rad]
            "left_thigh_joint": 0.0,
            "left_calf_joint": 0.0,
            "right_thigh_joint": 0.0,
            "right_calf_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        "joint_init_angles": {  # [rad]
            "left_thigh_joint": 0.0,
            "left_calf_joint": 0.0,
            "right_thigh_joint": 0.0,
            "right_calf_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        "dof_names": [
            "left_thigh_joint",  # 大腿
            "left_calf_joint",    # 小腿
            "right_thigh_joint",
            "right_calf_joint",
            "left_wheel_joint",   # 轮子
            "right_wheel_joint",
        ],
        # lower upper
        # 物理引擎的配置优先于urdf的配置
        # 由于wheel的joint设置是continuous，而其他关节是revolution。所以肯wheel的关节没有角度的限制，但是需要有速度限制和扭矩限制
        "dof_limit": {
            "left_thigh_joint": [0.0, 2.4435],  # 左大腿关节：垂直时为1.2217弧度(70°)，前后可摆动70°(1.2217弧度)
            "left_calf_joint": [-1.663, -0.082],  # 左小腿关节：大小腿一条直线时为-1.663弧度(-95.3°)，逆时针可旋转100°(1.7453弧度)至-0.082弧度(4.7°)
            "right_thigh_joint": [0.0, 2.4435],
            "right_calf_joint": [-1.663, -0.082],   #
            "left_wheel_joint": [0.0, 0.0],  # 固定
            "right_wheel_joint": [0.0, 0.0],  # 疑惑！
        },
        "safe_force": {
            "left_thigh_joint": 150.0,   # 左大腿关节最大扭矩是20N·M，增大数值可以输出更大的力，可以快速运动，但是可能损坏电机或者结构
            "left_calf_joint": 150.0,
            "right_thigh_joint": 150.0,
            "right_calf_joint": 150.0,
            "left_wheel_joint": 10.0, #给大点，测试的时候可以小
            "right_wheel_joint": 10.0,
        },
        # PD
        # 先增大 Kp，直到系统出现轻微震荡。再增大 Kv，使震荡平息。

        "joint_kp": 15,  # 对位置误差进行放大。较大的 Kp 能让系统响应更迅速，不过可能引发震荡。若 Kp过小，关节动作会变得迟缓；若过大，则可能导致关节剧烈抖动
        "joint_kv": 1.2,  # 对误差变化率起到阻尼作用，可抑制震荡，若 Kv​ 过小，系统可能会持续震荡；若过大，系统响应会变得缓慢
        "wheel_kv": 1.5,
        "damping": 0.005,  # 模拟关节的黏性阻力，其效果和 Kv类似，但它是物理层面的属性，较小的阻尼系数会使关节运动更灵活
        "stiffness":0.0, #不包含轮。  # 关节刚度。定义关节的弹性特性，值为 0 表示关节是刚性的
        "armature":0.004,  # 电机惯性。模拟电机转子的惯性，数值越小，电机响应越迅速。需要与电机参数匹配
        # termination 角度制    obs的angv弧度制
        # 终止条件的参数
        "termination_if_roll_greater_than": 20,  # degree  当机器人基座的侧倾角超过 20 度时，回合终止
        "termination_if_pitch_greater_than": 25, #15度以内都摆烂，会导致episode太短难以学习
        # "termination_if_base_height_greater_than": 0.1,
        # "termination_if_knee_height_greater_than": 0.00,
        "termination_if_base_connect_plane_than": True, #触地重置
        "connect_plane_links":[ #触地重置link
            "base_link",
            "left_knee_link",
            "left_hip_link",
            "right_knee_link",
            "right_hip_link",
                ],
        # base pose
        "base_init_pos":{
            "urdf":[0.0, 0.0, 0.35],#稍微高一点点
            "mjcf":[0.0, 0.0, 0.32],
            },
        # 机器人基座的初始姿态
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],#0.996195, 0, 0.0871557, 0， 这里是完全数值，没有旋转。初始姿态影响学习难度（如倾斜姿态增加平衡挑战）。sim2real需要与仿真有一样的初始姿态
        "episode_length_s": 20.0,  # 每个训练回合的最大持续时间（秒）。过短容易重启，过大容易无效探索
        "resampling_time_s": 3.0,
        "joint_action_scale": 0.5,
        "wheel_action_scale": 10,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        # num_obs = num_slice_obs + history_num * num_slice_obs
        "num_obs": 174, #在rsl-rl中使用的变量为num_obs表示state数量
        "num_slice_obs": 29,
        "history_length": 5,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.5,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
        },
        "noise":{
            "use": True,
            #[高斯,随机游走]
            "ang_vel": [0.01,0.1],
            "dof_pos": [0.02,0.01],
            "dof_vel": [0.01,0.1],
            "gravity": [0.02,0.02],
        }
    }
    # 名字和奖励函数名一一对应
    reward_cfg = {
        "tracking_lin_sigma": 0.6, 
        "tracking_ang_sigma": 0.5, 
        "tracking_height_sigma": 0.1,
        "tracking_similar_legged_sigma": 0.2,
        "tracking_gravity_sigma": 0.005,
        "tracking_pitch_sigma": 0.3,
        "reward_scales": {
            "tracking_lin_vel": 3.0,
            "tracking_ang_vel": 3.0,
            "tracking_base_height": 1.0,    #和similar_legged对抗，similar_legged先提升会促进此项
            "lin_vel_z": -0.21, #大了影响高度变换速度
            "joint_action_rate": -0.01,
            "wheel_action_rate": -0.005,
            "similar_to_default": 2.0,
            "projected_gravity": 6.0,
            "similar_legged": 0.7,  #tracking_base_height和knee_height对抗
            "dof_vel": -0.005,
            "dof_acc": -0.5e-9,
            "dof_force": -0.0001,
            "knee_height": -0.0,    #相当有效，和similar_legged结合可以抑制劈岔和跪地重启，稳定运行
            "ang_vel_xy": -0.02,
            "collision": -0.0008,  #base接触地面碰撞力越大越惩罚，数值太大会摆烂
            "pitch": 0.5,  #
            # "terrain":0.1,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "base_range": 1,  #基础范围
        "lin_vel_x_range": [-1.2, 1.2], #修改范围要调整奖励权重
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-3.14, 3.14],   #修改范围要调整奖励权重
        "height_target_range": [0.25, 0.25],   #lower会导致跪地
        #旋转和线速度范围会互冲，所以下面参数是降低互冲的（大概就行，轮不能完全代表整机），纯腿的用不上这个
        "limit_cmd_random":False,
        "wheel_spacing":0.37, #轮间距 m
        "wheel_radius":0.75, #轮半径 m
        "wheel_max_w":10, #轮最大转速 rad/s
    }
    # 课程学习，奖励循序渐进 待优化
    curriculum_cfg = {
        "curriculum_lin_vel_step":0.001,   #比例
        "curriculum_ang_vel_step":0.0003,   #比例
        "curriculum_height_target_step":0.0003,   #高度
        "curriculum_lin_vel_min_range":0.3,   #比例
        "curriculum_ang_vel_min_range":0.3,   #比例
        "lin_vel_err_range":[0.05,0.5],  #课程误差阈值
        "ang_vel_err_range":[0.20,0.45],  #课程误差阈值 连续曲线>方波>不波动
        "damping_descent":True,
        "dof_damping_descent":[0.2, 0.005, 0.001, 0.4],#[damping_max,damping_min,damping_step（比例）,damping_threshold（存活步数比例）]
    }
    #域随机化 friction_ratio是范围波动 mass和com是偏移波动
    domain_rand_cfg = { 
        "friction_ratio_range":[0.2 , 2.0],
        "random_base_mass_shift_range":[-2 , 2], #质量偏移量
        "random_other_mass_shift_range":[-0.1, 0.1],  #质量偏移量
        "random_base_com_shift":0.1, #位置偏移量 xyz
        "random_other_com_shift":0.01, #位置偏移量 xyz
        "random_KP":[0.8, 1.2], #比例
        "random_KV":[0.8, 1.2], #比例
        "random_default_joint_angles":[-0.05,0.05], #rad
        "damping_range":[0.9, 1.1], #比例
        "dof_stiffness_range":[0.0 , 0.0], #范围 不包含轮 [0.0 , 0.0]就是关闭，关闭的时候把初始值也调0
        "dof_stiffness_descent":[0.0 , 0.8], #刚度下降[max_stiffness，仿真步数占比阈值]，和域随机化冲突，二选一
        "dof_armature_range":[0.0 , 0.008], #范围 额外惯性 类似电机减速器惯性 有助于仿真稳定性
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking-v20000")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=25000)
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


# 若 Kv 和 damping 都设置得较大，关节可能会变得反应迟钝。
# 建议先固定 damping（取较小值），然后对 Kp 和 Kv 进行调参
# 终止条件与训练效率：
# 严格的终止条件（如小角度阈值）会使训练频繁重启。
# 宽松的终止条件（如大角度阈值）可能会让机器人学到不良姿态