"""
配置加载工具 - 统一管理YAML配置文件
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from .path_utils import paths


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """
    加载YAML文件
    
    Args:
        filepath: YAML文件路径
        
    Returns:
        Dict: 配置字典
    """
    if not filepath.exists():
        raise FileNotFoundError(f"配置文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], filepath: Path):
    """
    保存数据到YAML文件
    
    Args:
        data: 要保存的数据
        filepath: 保存路径
    """
    filepath.parent.mkdir(exist_ok=True, parents=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_config(config_type: str, config_name: str) -> Dict[str, Any]:
    """
    加载指定类型的配置文件
    
    Args:
        config_type: 配置类型 (robots, terrains, experiments)
        config_name: 配置名称
        
    Returns:
        Dict: 配置字典
    """
    config_path = paths.get_config_path(config_type, config_name)
    return load_yaml(config_path)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        Dict: 合并后的配置
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并字典
            result[key] = merge_configs(result[key], value)
        else:
            # 覆盖或添加新键
            result[key] = value
    
    return result


def load_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    加载实验配置（合并多个配置文件）
    
    Args:
        experiment_name: 实验名称
        
    Returns:
        Dict: 完整的实验配置
    """
    # 加载实验基础配置
    experiment_config = load_config("experiments", experiment_name)
    
    # 加载机器人配置
    robot_name = experiment_config.get("robot", {}).get("name", "wheel_leg_urdf")
    robot_config = load_config("robots", robot_name)
    
    # 加载地形配置
    terrain_name = experiment_config.get("terrain", {}).get("name", "flat")
    terrain_config = load_config("terrains", terrain_name)
    
    # 合并所有配置
    config = {}
    config = merge_configs(config, robot_config)
    config = merge_configs(config, terrain_config)
    config = merge_configs(config, experiment_config)
    
    return config


def create_default_configs():
    """创建默认配置文件"""
    
    # 创建默认机器人配置
    wheel_leg_urdf = {
        "robot": {
            "name": "wheel_leg_urdf",
            "type": "urdf",
            "file": "wheel_leg.urdf",
            "init_pos": [0.0, 0.0, 0.3],
            "init_quat": [1.0, 0.0, 0.0, 0.0],
            "joints": {
                "left_hip": "L1_joint",
                "left_knee": "L2_joint",
                "right_hip": "R1_joint",
                "right_knee": "R2_joint",
                "left_wheel": "L3_joint",
                "right_wheel": "R3_joint"
            }
        }
    }
    
    wheel_leg_mjcf = {
        "robot": {
            "name": "wheel_leg_mjcf",
            "type": "mjcf",
            "file": "wheel_leg.xml",
            "init_pos": [0.0, 0.0, 0.5],
            "init_quat": [1.0, 0.0, 0.0, 0.0]
        }
    }
    
    # 创建默认地形配置
    flat_terrain = {
        "terrain": {
            "name": "flat",
            "file": "plane.png",
            "horizontal_scale": 0.1,
            "vertical_scale": 0.0,
            "type": "flat"
        }
    }
    
    wave_terrain = {
        "terrain": {
            "name": "wave",
            "file": "terrain_wave.png",
            "horizontal_scale": 0.02,
            "vertical_scale": 0.001,
            "type": "wave"
        }
    }
    
    slope_10deg = {
        "terrain": {
            "name": "slope_10deg",
            "file": "slope_10deg_fit.png",
            "horizontal_scale": 0.05,
            "vertical_scale": 0.004,
            "type": "slope"
        }
    }
    
    # 创建默认实验配置
    fz_pid_experiment = {
        "experiment": {
            "name": "fz_pid",
            "description": "模糊PID控制实验",
            "version": "v0.1.9"
        },
        "training": {
            "algorithm": "fz_pid",
            "num_envs": 1,
            "max_steps": 10000,
            "save_interval": 100
        },
        "controller": {
            "type": "fuzzy_pid",
            "params": {
                "pitch_kp": 175,
                "pitch_ki": 0.3,
                "pitch_kd": 1.3,
                "vel_kp": 0.5,
                "vel_ki": 0.05,
                "vel_kd": 0
            }
        },
        "robot": {
            "name": "wheel_leg_urdf"
        },
        "terrain": {
            "name": "wave"
        }
    }
    
    basic_rl_experiment = {
        "experiment": {
            "name": "basic_rl",
            "description": "基础强化学习实验",
            "version": "v0.0.1"
        },
        "training": {
            "algorithm": "ppo",
            "num_envs": 1,
            "max_steps": 10000,
            "save_interval": 100
        },
        "robot": {
            "name": "wheel_leg_urdf"
        },
        "terrain": {
            "name": "flat"
        }
    }
    
    sim2sim_experiment = {
        "experiment": {
            "name": "sim2sim",
            "description": "Sim2Sim对比实验",
            "version": "v0.0.7"
        },
        "training": {
            "algorithm": "fz_pid",
            "num_envs": 1,
            "max_steps": 5000,
            "save_interval": 50
        },
        "controller": {
            "type": "fuzzy_pid",
            "params": {
                "pitch_kp": 175,
                "pitch_ki": 0.3,
                "pitch_kd": 1.3,
                "vel_kp": 0.5,
                "vel_ki": 0.05,
                "vel_kd": 0
            }
        },
        "robot": {
            "name": "wheel_leg_urdf"
        },
        "terrain": {
            "name": "flat"
        },
        "sim2sim": {
            "enabled": True,
            "mujoco_comparison": True
        }
    }
    
    # 保存所有配置
    configs = [
        ("robots", "wheel_leg_urdf", wheel_leg_urdf),
        ("robots", "wheel_leg_mjcf", wheel_leg_mjcf),
        ("terrains", "flat", flat_terrain),
        ("terrains", "wave", wave_terrain),
        ("terrains", "slope_10deg", slope_10deg),
        ("experiments", "fz_pid", fz_pid_experiment),
        ("experiments", "basic_rl", basic_rl_experiment),
        ("experiments", "sim2sim", sim2sim_experiment),
    ]
    
    for config_type, config_name, config_data in configs:
        config_path = paths.configs / config_type / f"{config_name}.yaml"
        save_yaml(config_data, config_path)
        print(f"✅ 创建配置: {config_type}/{config_name}.yaml")


if __name__ == "__main__":
    # 测试配置加载
    print("🔍 测试配置加载工具...")
    create_default_configs()
    
    # 加载实验配置
    config = load_experiment_config("fz_pid")
    print(f"实验配置加载成功，包含 {len(config)} 个顶级键")
    print(f"实验名称: {config.get('experiment', {}).get('name')}")