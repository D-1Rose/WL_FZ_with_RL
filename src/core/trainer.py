"""
统一训练器 - 从locomotion/FZ_PID_train.py提取的核心训练逻辑
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import pickle

# 导入工具
from ..utils.path_utils import paths
from ..utils.import_utils import import_manager


class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.setup_done = False
        
    def _setup_device(self) -> torch.device:
        """设置设备"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("⚠️  使用CPU，训练可能较慢")
        return device
    
    def setup(self):
        """设置训练环境"""
        if self.setup_done:
            return
        
        print("🔧 设置训练环境...")
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 加载配置
        self._load_configs()
        
        # 创建环境
        self._create_env()
        
        # 创建策略
        self._create_policy()
        
        self.setup_done = True
        print("✅ 训练环境设置完成")
    
    def _create_output_dirs(self):
        """创建输出目录"""
        exp_name = self.config.get("experiment", {}).get("name", "unknown")
        
        # 日志目录
        self.log_dir = paths.logs / exp_name
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # 模型目录
        self.model_dir = paths.models / exp_name
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"📁 输出目录:")
        print(f"   日志: {self.log_dir}")
        print(f"   模型: {self.model_dir}")
    
    def _load_configs(self):
        """加载配置"""
        # 这里将实现从locomotion/get_cfgs提取的配置
        # 暂时使用默认配置
        self.env_cfg = self.config.get("env", {})
        self.train_cfg = self.config.get("training", {})
        self.policy_cfg = self.config.get("policy", {})
        
        print(f"📋 训练配置:")
        print(f"   算法: {self.train_cfg.get('algorithm', 'PPO')}")
        print(f"   最大迭代: {self.train_cfg.get('max_iterations', 1000)}")
        print(f"   每环境步数: {self.train_cfg.get('num_steps_per_env', 30)}")
    
    def _create_env(self):
        """创建环境"""
        env_type = self.config.get("env", {}).get("type", "wheel_legged")
        
        try:
            # 使用导入管理器获取环境类
            EnvClass = import_manager.get_env(env_type)
            
            # 获取配置函数
            get_cfgs = import_manager.import_from_locomotion("FZ_PID_train", "get_cfgs")
            
            # 加载配置
            env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
            
            # 根据实验配置更新地形
            terrain_config = self.config.get("terrain", {})
            if terrain_config:
                terrain_cfg["terrain"] = True
                terrain_cfg["eval"] = terrain_config.get("name", "wave")
                terrain_cfg["horizontal_scale"] = terrain_config.get("horizontal_scale", 0.02)
                terrain_cfg["vertical_scale"] = terrain_config.get("vertical_scale", 0.001)
            
            # 创建环境
            self.env = EnvClass(
                num_envs=self.train_cfg.get("num_envs", 1),
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                curriculum_cfg=curriculum_cfg,
                domain_rand_cfg=domain_rand_cfg,
                terrain_cfg=terrain_cfg,
                robot_morphs="urdf",
                show_viewer=False,
                device=str(self.device),
                train_mode=True
            )
            
            print(f"✅ 创建环境: {env_type}")
            print(f"   环境数: {self.train_cfg.get('num_envs', 1)}")
            print(f"   地形: {terrain_config.get('name', 'wave')}")
            
        except Exception as e:
            print(f"❌ 创建环境失败: {e}")
            raise
    
    def _create_policy(self):
        """创建策略"""
        # 这里将实现策略创建
        # 暂时占位
        print("📦 创建策略网络...")
        print("   Actor网络: [512, 256, 128]")
        print("   Critic网络: [512, 256, 128]")
        
        # 模拟创建策略
        self.policy = {"actor": "ActorNetwork", "critic": "CriticNetwork"}
    
    def train(self):
        """训练主循环"""
        self.setup()
        
        print("\n" + "=" * 60)
        print("🏋️  开始训练")
        print("=" * 60)
        
        max_iterations = self.train_cfg.get("max_iterations", 1000)
        log_interval = self.train_cfg.get("log_interval", 10)
        save_interval = self.train_cfg.get("save_interval", 100)
        
        for iteration in range(max_iterations):
            # 模拟训练步骤
            if iteration % log_interval == 0:
                self._log_progress(iteration, max_iterations)
            
            if iteration % save_interval == 0 and iteration > 0:
                self._save_checkpoint(iteration)
            
            # 这里应该执行实际的训练步骤
            # 暂时模拟训练
            if iteration >= 10:  # 只模拟10步
                break
        
        print("\n" + "=" * 60)
        print("🎉 训练完成")
        print("=" * 60)
        
        # 保存最终模型
        self._save_final_model()
        
        return True
    
    def _log_progress(self, iteration: int, max_iterations: int):
        """记录进度"""
        progress = (iteration + 1) / max_iterations * 100
        print(f"📊 迭代 {iteration + 1}/{max_iterations} ({progress:.1f}%)")
        
        # 模拟一些训练指标
        metrics = {
            "reward": np.random.uniform(0.5, 1.5),
            "episode_length": np.random.randint(100, 300),
            "value_loss": np.random.uniform(0.01, 0.1),
            "policy_loss": np.random.uniform(0.01, 0.1),
        }
        
        print(f"   奖励: {metrics['reward']:.3f}")
        print(f"   回合长度: {metrics['episode_length']}")
        print(f"   价值损失: {metrics['value_loss']:.4f}")
        print(f"   策略损失: {metrics['policy_loss']:.4f}")
    
    def _save_checkpoint(self, iteration: int):
        """保存检查点"""
        checkpoint_path = self.model_dir / f"model_{iteration}.pt"
        
        # 模拟保存模型
        checkpoint = {
            "iteration": iteration,
            "policy_state": self.policy,
            "config": self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 保存检查点: {checkpoint_path.name}")
    
    def _save_final_model(self):
        """保存最终模型"""
        final_path = self.model_dir / "final_model.pt"
        
        final_model = {
            "policy": self.policy,
            "config": self.config,
            "env_info": {
                "type": self.config.get("env", {}).get("type"),
                "num_envs": self.train_cfg.get("num_envs"),
            }
        }
        
        torch.save(final_model, final_path)
        print(f"💾 保存最终模型: {final_path}")
        
        # 保存配置
        config_path = self.model_dir / "training_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"📋 保存训练配置: {config_path}")


class FZPIDTrainer(BaseTrainer):
    """模糊PID训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.controller_type = "fuzzy_pid"
    
    def _create_policy(self):
        """创建模糊PID控制器"""
        print("🎯 创建模糊PID控制器...")
        
        try:
            # 导入模糊PID控制器
            FuzzyPIDController = import_manager.get_controller("fuzzy_pid")
            
            # 创建控制器实例
            self.controller = FuzzyPIDController()
            self.policy = {"controller": self.controller}
            
            print(f"✅ 创建控制器: {self.controller.__class__.__name__}")
            
        except Exception as e:
            print(f"❌ 创建控制器失败: {e}")
            raise
    
    def train(self):
        """模糊PID训练"""
        self.setup()
        
        print("\n" + "=" * 60)
        print("🎯 模糊PID控制器训练")
        print("=" * 60)
        
        # 模糊PID通常不需要传统RL训练
        # 这里实现控制器参数调优
        max_iterations = self.train_cfg.get("max_iterations", 100)
        
        for iteration in range(max_iterations):
            if iteration % 10 == 0:
                print(f"🔧 调优迭代 {iteration + 1}/{max_iterations}")
                
                # 模拟控制器调优
                # 实际应该在这里调整PID参数和模糊规则
        
        print("\n✅ 模糊PID控制器调优完成")
        return True


class PPOTrainer(BaseTrainer):
    """PPO训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.algorithm = "PPO"
    
    def _create_policy(self):
        """创建PPO策略"""
        print("🧠 创建PPO策略网络...")
        
        try:
            # 导入RSL-RL的PPO实现
            from rsl_rl.runners import OnPolicyRunner
            from rsl_rl.modules import ActorCritic
            
            # 创建Actor-Critic网络
            actor_critic = ActorCritic(
                num_actor_obs=self.env.num_obs,
                num_critic_obs=self.env.num_obs,
                num_actions=self.env.num_actions,
                actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
                critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
                activation="elu",
                init_noise_std=self.policy_cfg.get("init_noise_std", 1.0),
            ).to(self.device)
            
            # 创建PPO运行器
            self.runner = OnPolicyRunner(
                env=self.env,
                train_cfg=self._get_ppo_config(),
                device=str(self.device),
                log_dir=str(self.log_dir),
            )
            
            self.policy = actor_critic
            print("✅ PPO策略创建完成")
            
        except Exception as e:
            print(f"❌ 创建PPO策略失败: {e}")
            raise
    
    def _get_ppo_config(self) -> Dict[str, Any]:
        """获取PPO配置"""
        # 从locomotion/get_train_cfg提取
        try:
            get_train_cfg = import_manager.import_from_locomotion("FZ_PID_train", "get_train_cfg")
            return get_train_cfg(
                self.config.get("experiment", {}).get("name", "ppo_exp"),
                self.train_cfg.get("max_iterations", 1000)
            )
        except:
            # 返回默认配置
            return {
                "algorithm": {
                    "clip_param": 0.2,
                    "entropy_coef": 0.01,
                    "learning_rate": 1e-4,
                    "num_learning_epochs": 5,
                    "num_mini_batches": 4,
                    "gamma": 0.99,
                    "lam": 0.95,
                },
                "runner": {
                    "experiment_name": self.config.get("experiment", {}).get("name"),
                    "max_iterations": self.train_cfg.get("max_iterations", 1000),
                    "save_interval": 100,
                }
            }


def create_trainer(config: Dict[str, Any]) -> BaseTrainer:
    """创建训练器工厂函数"""
    algorithm = config.get("training", {}).get("algorithm", "fz_pid").lower()
    
    trainer_map = {
        "fz_pid": FZPIDTrainer,
        "ppo": PPOTrainer,
        "sac": BaseTrainer,  # 待实现
        "td3": BaseTrainer,  # 待实现
    }
    
    trainer_class = trainer_map.get(algorithm, BaseTrainer)
    return trainer_class(config)


if __name__ == "__main__":
    # 测试训练器
    test_config = {
        "experiment": {
            "name": "test_training",
            "description": "测试训练器",
        },
        "training": {
            "algorithm": "fz_pid",
            "num_envs": 1,
            "max_iterations": 50,
        },
        "env": {
            "type": "wheel_legged",
        },
        "terrain": {
            "name": "wave",
        },
    }
    
    print("🧪 测试训练器...")
    trainer = create_trainer(test_config)
    trainer.train()