"""
统一评估器 - 从locomotion/FZ_PID_eval.py提取的核心评估逻辑
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

# 导入工具
from ..utils.path_utils import paths
from ..utils.import_utils import import_manager


class BaseEvaluator:
    """基础评估器"""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[Path] = None):
        self.config = config
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def setup(self):
        """设置评估环境"""
        print("🔧 设置评估环境...")
        
        # 加载模型
        self._load_model()
        
        # 创建环境
        self._create_env()
        
        # 创建策略/控制器
        self._create_policy()
        
        print("✅ 评估环境设置完成")
    
    def _load_model(self):
        """加载模型"""
        if self.model_path and self.model_path.exists():
            print(f"📦 加载模型: {self.model_path}")
            try:
                self.model = torch.load(self.model_path, map_location=self.device)
                print(f"✅ 模型加载成功")
                
                # 提取模型信息
                if isinstance(self.model, dict):
                    self.model_info = self.model.get("config", {})
                    self.policy_state = self.model.get("policy_state", {})
                else:
                    self.model_info = {}
                    self.policy_state = self.model.state_dict() if hasattr(self.model, "state_dict") else {}
                    
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model = None
        else:
            print("⚠️  未提供模型路径，使用随机策略")
            self.model = None
    
    def _create_env(self):
        """创建评估环境"""
        env_type = self.config.get("env", {}).get("type", "wheel_legged")
        
        try:
            # 使用导入管理器获取环境类
            EnvClass = import_manager.get_env(env_type)
            
            # 获取配置函数
            get_cfgs = import_manager.import_from_locomotion("FZ_PID_train", "get_cfgs")
            
            # 加载配置
            env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
            
            # 评估模式使用固定地形
            terrain_config = self.config.get("terrain", {})
            terrain_cfg["terrain"] = True
            terrain_cfg["eval"] = terrain_config.get("name", "wave")
            terrain_cfg["horizontal_scale"] = terrain_config.get("horizontal_scale", 0.02)
            terrain_cfg["vertical_scale"] = terrain_config.get("vertical_scale", 0.001)
            
            # 创建环境（评估模式）
            self.env = EnvClass(
                num_envs=1,  # 评估通常使用单个环境
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                curriculum_cfg=curriculum_cfg,
                domain_rand_cfg=domain_rand_cfg,
                terrain_cfg=terrain_cfg,
                robot_morphs="urdf",
                show_viewer=False,  # 评估时关闭可视化
                device=str(self.device),
                train_mode=False  # 评估模式
            )
            
            print(f"✅ 创建评估环境: {env_type}")
            print(f"   地形: {terrain_config.get('name', 'wave')}")
            
        except Exception as e:
            print(f"❌ 创建环境失败: {e}")
            raise
    
    def _create_policy(self):
        """创建策略"""
        algorithm = self.config.get("training", {}).get("algorithm", "fz_pid")
        
        if algorithm == "fz_pid":
            self._create_fz_pid_policy()
        elif algorithm == "ppo":
            self._create_ppo_policy()
        else:
            print(f"⚠️  未知算法 {algorithm}，使用随机策略")
            self.policy = None
    
    def _create_fz_pid_policy(self):
        """创建模糊PID策略"""
        print("🎯 创建模糊PID控制器...")
        
        try:
            # 导入模糊PID控制器
            FuzzyPIDController = import_manager.get_controller("fuzzy_pid")
            
            # 创建控制器实例
            self.controller = FuzzyPIDController()
            
            # 如果模型中有控制器参数，加载它们
            if self.policy_state and "controller" in self.policy_state:
                self.controller.load_state_dict(self.policy_state["controller"])
                print("✅ 加载控制器参数")
            
            self.policy = self.controller
            print(f"✅ 控制器: {self.controller.__class__.__name__}")
            
        except Exception as e:
            print(f"❌ 创建控制器失败: {e}")
            self.policy = None
    
    def _create_ppo_policy(self):
        """创建PPO策略"""
        print("🧠 创建PPO策略...")
        
        try:
            from rsl_rl.modules import ActorCritic
            
            # 创建Actor-Critic网络
            self.policy = ActorCritic(
                num_actor_obs=self.env.num_obs,
                num_critic_obs=self.env.num_obs,
                num_actions=self.env.num_actions,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation="elu",
                init_noise_std=1.0,
            ).to(self.device)
            
            # 加载模型参数
            if self.policy_state:
                self.policy.load_state_dict(self.policy_state)
                print("✅ 加载策略参数")
            
            print("✅ PPO策略创建完成")
            
        except Exception as e:
            print(f"❌ 创建PPO策略失败: {e}")
            self.policy = None
    
    def evaluate(self, num_episodes: int = 10, max_steps: int = 1000) -> Dict[str, Any]:
        """运行评估"""
        self.setup()
        
        print("\n" + "=" * 60)
        print("📊 开始评估")
        print("=" * 60)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            print(f"\\n📈 评估回合 {episode + 1}/{num_episodes}")
            
            # 重置环境
            obs = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done and step < max_steps:
                # 获取动作
                action = self._get_action(obs)
                
                # 执行动作
                obs, reward, done, info = self.env.step(action)
                
                total_reward += reward
                step += 1
                
                # 显示进度
                if step % 100 == 0:
                    print(f"   步数: {step}, 累计奖励: {total_reward:.2f}")
            
            episode_rewards.append(total_reward)
            episode_lengths.append(step)
            
            print(f"✅ 回合完成: 奖励={total_reward:.2f}, 步数={step}")
        
        # 计算统计信息
        self.results = self._compute_statistics(episode_rewards, episode_lengths)
        
        # 保存结果
        self._save_results()
        
        # 可视化结果
        self._visualize_results(episode_rewards, episode_lengths)
        
        print("\n" + "=" * 60)
        print("🎉 评估完成")
        print("=" * 60)
        
        return self.results
    
    def _get_action(self, obs):
        """获取动作"""
        if self.policy is None:
            # 随机策略
            return np.random.uniform(-1, 1, size=self.env.num_actions)
        
        algorithm = self.config.get("training", {}).get("algorithm", "fz_pid")
        
        if algorithm == "fz_pid":
            # 模糊PID控制器
            return self.policy(obs)
        elif algorithm == "ppo":
            # PPO策略
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                action = self.policy.act_inference(obs_tensor)
                return action.cpu().numpy()[0]
        else:
            # 默认随机
            return np.random.uniform(-1, 1, size=self.env.num_actions)
    
    def _compute_statistics(self, rewards: List[float], lengths: List[int]) -> Dict[str, Any]:
        """计算统计信息"""
        rewards_array = np.array(rewards)
        lengths_array = np.array(lengths)
        
        return {
            "mean_reward": float(np.mean(rewards_array)),
            "std_reward": float(np.std(rewards_array)),
            "min_reward": float(np.min(rewards_array)),
            "max_reward": float(np.max(rewards_array)),
            "mean_length": float(np.mean(lengths_array)),
            "std_length": float(np.std(lengths_array)),
            "success_rate": float(np.mean(lengths_array > 100)),  # 假设步数>100为成功
            "num_episodes": len(rewards),
        }
    
    def _save_results(self):
        """保存评估结果"""
        exp_name = self.config.get("experiment", {}).get("name", "unknown")
        results_dir = paths.logs / exp_name / "eval_results"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存为JSON
        import json
        results_file = results_dir / "evaluation_results.json"
        
        results_data = {
            "config": self.config,
            "results": self.results,
            "model_path": str(self.model_path) if self.model_path else None,
            "timestamp": np.datetime64('now').astype(str),
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 保存评估结果: {results_file}")
        
        # 保存为文本报告
        report_file = results_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\\n")
            f.write("📊 评估报告\\n")
            f.write("=" * 60 + "\\n\\n")
            
            f.write(f"实验: {exp_name}\\n")
            f.write(f"算法: {self.config.get('training', {}).get('algorithm', 'unknown')}\\n")
            f.write(f"地形: {self.config.get('terrain', {}).get('name', 'unknown')}\\n")
            f.write(f"模型: {self.model_path.name if self.model_path else '无'}\\n\\n")
            
            f.write("📈 性能指标:\\n")
            f.write(f"   平均奖励: {self.results['mean_reward']:.2f} ± {self.results['std_reward']:.2f}\\n")
            f.write(f"   奖励范围: [{self.results['min_reward']:.2f}, {self.results['max_reward']:.2f}]\\n")
            f.write(f"   平均步数: {self.results['mean_length']:.1f} ± {self.results['std_length']:.1f}\\n")
            f.write(f"   成功率: {self.results['success_rate']*100:.1f}%\\n")
        
        print(f"📋 保存评估报告: {report_file}")
    
    def _visualize_results(self, rewards: List[float], lengths: List[int]):
        """可视化结果"""
        try:
            exp_name = self.config.get("experiment", {}).get("name", "unknown")
            results_dir = paths.logs / exp_name / "eval_results"
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 奖励分布
            axes[0, 0].plot(rewards, 'b-o', alpha=0.7)
            axes[0, 0].axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'平均: {np.mean(rewards):.2f}')
            axes[0, 0].set_xlabel('回合')
            axes[0, 0].set_ylabel('奖励')
            axes[0, 0].set_title('奖励随回合变化')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 步数分布
            axes[0, 1].plot(lengths, 'g-o', alpha=0.7)
            axes[0, 1].axhline(y=np.mean(lengths), color='r', linestyle='--', label=f'平均: {np.mean(lengths):.1f}')
            axes[0, 1].set_xlabel('回合')
            axes[0, 1].set_ylabel('步数')
            axes[0, 1].set_title('步数随回合变化')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 奖励直方图
            axes[1, 0].hist(rewards, bins=10, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].axvline(x=np.mean(rewards), color='r', linestyle='--', label=f'平均: {np.mean(rewards):.2f}')
            axes[1, 0].set_xlabel('奖励')
            axes[1, 0].set_ylabel('频率')
            axes[1, 0].set_title('奖励分布')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 汇总统计
            axes[1, 1].axis('off')
            stats_text = f"""
            评估统计:
            ============
            回合数: {len(rewards)}
            平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}
            奖励范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]
            平均步数: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}
            成功率: {(np.mean(np.array(lengths) > 100)*100):.1f}%
            """
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
            
            plt.suptitle(f'{exp_name} - 评估结果', fontsize=14)
            plt.tight_layout()
            
            # 保存图形
            plot_file = results_dir / "evaluation_plots.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 保存可视化结果: {plot_file}")
            
        except Exception as e:
            print(f"⚠️  可视化失败: {e}")


def create_evaluator(config: Dict[str, Any], model_path: Optional[Path] = None) -> BaseEvaluator:
    """创建评估器工厂函数"""
    return BaseEvaluator(config, model_path)


if __name__ == "__main__":
    # 测试评估器
    test_config = {
        "experiment": {
            "name": "test_evaluation",
            "description": "测试评估器",
        },
        "training": {
            "algorithm": "fz_pid",
        },
        "env": {
            "type": "wheel_legged",
        },
        "terrain": {
            "name": "wave",
        },
    }
    
    print("🧪 测试评估器...")
    evaluator = create_evaluator(test_config)
    results = evaluator.evaluate(num_episodes=3, max_steps=200)
    
    print(f"\\n📊 评估结果:")
    for key, value in results.items():
        print(f"  {key}: {value}")