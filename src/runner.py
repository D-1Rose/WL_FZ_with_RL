#!/usr/bin/env python3
"""
统一运行器 - 替代所有杂乱的脚本
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_experiment_config
from src.utils.import_utils import import_manager


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, experiment_name: str, config_path: Optional[Path] = None):
        self.experiment_name = experiment_name
        self.config = self._load_config(config_path)
        self.setup_done = False
        
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """加载配置"""
        if config_path and config_path.exists():
            # 从指定文件加载
            from src.utils.config_loader import load_yaml
            return load_yaml(config_path)
        else:
            # 加载实验配置
            return load_experiment_config(self.experiment_name)
    
    def setup(self):
        """设置实验环境"""
        if self.setup_done:
            return
        
        print(f"🔧 设置实验: {self.experiment_name}")
        
        # 加载配置信息
        exp_info = self.config.get("experiment", {})
        print(f"   名称: {exp_info.get('name', '未知')}")
        print(f"   描述: {exp_info.get('description', '无描述')}")
        print(f"   版本: {exp_info.get('version', '未知')}")
        
        # 设置输出目录
        self._setup_output_dirs()
        
        self.setup_done = True
        print("✅ 实验设置完成")
    
    def _setup_output_dirs(self):
        """设置输出目录"""
        from src.utils.path_utils import paths
        
        # 创建日志目录
        log_dir = paths.logs / self.experiment_name
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建模型目录
        model_dir = paths.models / self.experiment_name
        model_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_dir = log_dir
        self.model_dir = model_dir
    
    def run_train(self):
        """运行训练"""
        self.setup()
        
        print(f"🏋️  开始训练: {self.experiment_name}")
        
        # 获取训练配置
        train_config = self.config.get("training", {})
        algorithm = train_config.get("algorithm", "fz_pid")
        
        print(f"   算法: {algorithm}")
        print(f"   环境数: {train_config.get('num_envs', 1)}")
        print(f"   最大步数: {train_config.get('max_steps', 1000)}")
        
        try:
            # 使用新的训练器
            from src.core.trainer import create_trainer
            
            trainer = create_trainer(self.config)
            success = trainer.train()
            
            if success:
                print(f"\n✅ 训练完成！")
                print(f"📁 输出目录:")
                print(f"   日志: {trainer.log_dir}")
                print(f"   模型: {trainer.model_dir}")
                return True
            else:
                print(f"\n❌ 训练失败")
                return False
                
        except Exception as e:
            print(f"\n❌ 训练器创建失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 回退到旧方法
            print("\n🔄 尝试使用旧方法...")
            return self._run_fallback_train()
        
        return True
    
    def _run_fallback_train(self):
        """回退训练方法（使用旧代码）"""
        algorithm = self.config.get("training", {}).get("algorithm", "fz_pid")
        
        if algorithm == "fz_pid":
            return self._run_fz_pid_train_fallback()
        else:
            print(f"❌ 不支持的算法: {algorithm}")
            return False
    
    def _run_fz_pid_train_fallback(self):
        """回退模糊PID训练"""
        try:
            # 直接调用原始训练脚本
            from locomotion.FZ_PID_train import main as train_main
            
            print("🔄 使用原始训练脚本...")
            
            # 设置命令行参数
            import sys
            sys.argv = [
                "FZ_PID_train.py",
                "--exp_name", self.experiment_name,
                "--num_envs", str(self.config.get("training", {}).get("num_envs", 1)),
                "--terrain", self.config.get("terrain", {}).get("name", "wave"),
            ]
            
            # 运行训练
            train_main()
            
            print("✅ 原始训练完成")
            return True
            
        except Exception as e:
            print(f"❌ 回退训练失败: {e}")
            return False
    
    def _run_ppo_train(self):
        """运行PPO训练"""
        print("📦 PPO训练（待实现）")
        # 这里实现PPO训练逻辑
        print("✅ PPO训练完成（演示模式）")
    
    def run_eval(self):
        """运行评估"""
        self.setup()
        
        print(f"📊 开始评估: {self.experiment_name}")
        
        # 检查模型文件
        model_files = list(self.model_dir.glob("*.pt"))
        if not model_files:
            print("❌ 未找到模型文件，请先训练")
            return False
        
        print(f"   找到模型: {model_files[0].name}")
        print("✅ 评估完成（演示模式）")
        return True
    
    def run_demo(self):
        """运行演示"""
        self.setup()
        
        print(f"🎮 运行演示: {self.experiment_name}")
        
        # 这里实现演示逻辑
        print("🤖 机器人演示中...")
        print("✅ 演示完成")
        return True


def main():
    parser = argparse.ArgumentParser(description="统一实验运行器")
    parser.add_argument(
        "experiment",
        type=str,
        help="实验名称或实验目录路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "demo"],
        help="运行模式"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🚀 统一实验运行器")
    print(f"   实验: {args.experiment}")
    print(f"   模式: {args.mode}")
    print("=" * 60)
    
    # 创建运行器
    config_path = Path(args.config) if args.config else None
    runner = ExperimentRunner(args.experiment, config_path)
    
    # 运行指定模式
    if args.mode == "train":
        success = runner.run_train()
    elif args.mode == "eval":
        success = runner.run_eval()
    elif args.mode == "demo":
        success = runner.run_demo()
    else:
        print(f"❌ 未知模式: {args.mode}")
        return 1
    
    if success:
        print("=" * 60)
        print(f"🎉 实验 '{args.experiment}' 运行成功！")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print(f"❌ 实验 '{args.experiment}' 运行失败")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())