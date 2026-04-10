#!/usr/bin/env python3
"""
文件索引和导入重定向系统
确保文件移动后原有导入仍然有效
"""

import os
import sys
import importlib
import importlib.util
import importlib.abc
from pathlib import Path
import warnings

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 文件映射表：旧路径 -> 新路径
# 格式: "old_module_path": "new_module_path"
FILE_MAPPINGS = {
    # locomotion/ 目录下的文件
    "locomotion.Controller": "locomotion.controllers.Controller",
    "locomotion.controller_text": "locomotion.controllers.controller_text",
    "locomotion.fuzzy_control": "locomotion.controllers.fuzzy_control",
    "locomotion.FZ_PID_c": "locomotion.controllers.FZ_PID_c",
    "FZ_PID_c": "locomotion.controllers.FZ_PID_c",  # 为FZ_PID_env_c.py添加的绝对导入映射
    "locomotion.Offset_mj_controller": "locomotion.controllers.Offset_mj_controller",
    "locomotion.Pitch_controller": "locomotion.controllers.Pitch_controller",
    
    "locomotion.FZ_PID_env_c": "locomotion.environments.FZ_PID_env_c",
    "FZ_PID_env_c": "locomotion.environments.FZ_PID_env_c",  # 为FZ_PID_eval.py添加的绝对导入映射
    "FZ_RL_ENV": "locomotion.environments.FZ_RL_ENV",
    "Pitch_env": "locomotion.environments.Pitch_env",
    "Controller": "locomotion.controllers.Controller",
    "Pitch_controller": "locomotion.controllers.Pitch_controller",
    "Offset_mj_controller": "locomotion.controllers.Offset_mj_controller",
    "FZ_PID_train": "locomotion.trainers.FZ_PID_train",
    "Sim_mj_keyboard": "locomotion.utils.Sim_mj_keyboard",
    "Sim_mj": "locomotion.environments.Sim_mj",
    "locomotion.FZ_RL_ENV": "locomotion.environments.FZ_RL_ENV",
    "locomotion.Pitch_env": "locomotion.environments.Pitch_env",
    "locomotion.mj": "locomotion.environments.mj",
    
    "locomotion.FZ_PID_train": "locomotion.trainers.FZ_PID_train",
    "locomotion.FZ_RL_TRAIN": "locomotion.trainers.FZ_RL_TRAIN",
    "locomotion.Pitch_train": "locomotion.trainers.Pitch_train",
    
    "locomotion.FZ_PID_eval": "locomotion.evaluators.FZ_PID_eval",
    "locomotion.FZ_RL_EVAL": "locomotion.evaluators.FZ_RL_EVAL",
    "locomotion.Pitch_eval": "locomotion.evaluators.Pitch_eval",
    "locomotion.mj_eval": "locomotion.evaluators.mj_eval",
    "locomotion.Offset_mj_eval": "locomotion.evaluators.Offset_mj_eval",
    
    "locomotion.data2": "locomotion.evaluators.mj_eval",  # 重命名为mj_eval
    "locomotion.mj_keyboard": "locomotion.utils.mj_keyboard",
    "mj_keyboard": "locomotion.utils.mj_keyboard",  # 为data2.py添加的绝对导入映射
    "locomotion.mj_pid_test": "locomotion.utils.mj_pid_test",
    "locomotion.num_envs_fz_control": "locomotion.utils.num_envs_fz_control",
    "locomotion.Offset_mj_plot": "locomotion.utils.Offset_mj_plot",
    "locomotion.Picture": "locomotion.utils.Picture",
    "locomotion.Pitch_genesis_pid_test": "locomotion.utils.Pitch_genesis_pid_test",
    "locomotion.SET2_FZ_picture": "locomotion.utils.SET2_FZ_picture",
    "locomotion.Sim_mj_keyboard": "locomotion.utils.Sim_mj_keyboard",
    "locomotion.Sim_mj_pid_test": "locomotion.utils.Sim_mj_pid_test",
    "locomotion.Sim_mj": "locomotion.environments.Sim_mj",
    "mj": "locomotion.environments.mj",  # 为data2.py添加的绝对导入映射
    "locomotion.terrain": "locomotion.utils.terrain",
    "locomotion.typ2_pid_vmc": "locomotion.controllers.typ2_pid_vmc",
    "locomotion.urdf_test": "locomotion.utils.urdf_test",
    "locomotion.vmc_lqr_keyboard": "locomotion.utils.vmc_lqr_keyboard",
    "locomotion.wheel_legged_env": "locomotion.environments.wheel_legged_env",
    "locomotion.wheel_legged_eval": "locomotion.evaluators.wheel_legged_eval",
    "locomotion.wheel_legged_train_our": "locomotion.trainers.wheel_legged_train_our",
    "locomotion.wheel_legged_train": "locomotion.trainers.wheel_legged_train",
    "locomotion.wl_fz_env": "locomotion.environments.wl_fz_env",
    "locomotion.wl_fz_eval": "locomotion.evaluators.wl_fz_eval",
    "locomotion.wl_fz_train": "locomotion.trainers.wl_fz_train",
    "locomotion.xml_test": "locomotion.utils.xml_test",
    
    # 根目录下的脚本文件
    "test_minimal": "scripts.test.test_minimal",
    "test_integration": "scripts.test.test_integration",
    "test_constraint_curve": "scripts.test.test_constraint_curve",
    "test_all_scripts": "scripts.test.test_all_scripts",
    "test_original_flow": "scripts.test.test_original_flow",
    
    "verify_restructure": "scripts.tools.verify_restructure",
    "test_refactor": "scripts.tools.test_refactor",
    "verify_success": "scripts.tools.verify_success",
}

# 反向映射：新路径 -> 旧路径（用于查找）
REVERSE_MAPPINGS = {v: k for k, v in FILE_MAPPINGS.items()}

class FileIndexFinder(importlib.abc.MetaPathFinder):
    """自定义导入查找器，处理文件重定向"""
    
    def find_spec(self, fullname, path, target=None):
        # 检查是否在映射表中
        if fullname in FILE_MAPPINGS:
            new_fullname = FILE_MAPPINGS[fullname]
            # 尝试导入新位置
            spec = importlib.util.find_spec(new_fullname)
            if spec is not None:
                # 创建重定向加载器
                spec.loader = FileIndexLoader(new_fullname, fullname)
                return spec
        
        # 检查是否是新位置，但需要警告
        if fullname in REVERSE_MAPPINGS:
            old_fullname = REVERSE_MAPPINGS[fullname]
            warnings.warn(
                f"导入 '{fullname}' 是新的模块路径。"
                f"考虑更新代码使用新路径，而不是旧路径 '{old_fullname}'。",
                DeprecationWarning,
                stacklevel=2
            )
        
        return None

class FileIndexLoader(importlib.abc.Loader):
    """自定义加载器，处理模块重定向"""
    
    def __init__(self, new_module_name, old_module_name):
        self.new_module_name = new_module_name
        self.old_module_name = old_module_name
    
    def create_module(self, spec):
        # 导入实际模块
        new_module = importlib.import_module(self.new_module_name)
        
        # 创建代理模块
        module = type(sys)(self.old_module_name)
        module.__dict__.update(new_module.__dict__)
        module.__file__ = new_module.__file__
        module.__loader__ = self
        module.__package__ = new_module.__package__
        
        # 添加重定向信息
        module.__redirected_from__ = self.old_module_name
        module.__redirected_to__ = self.new_module_name
        
        return module
    
    def exec_module(self, module):
        # 模块已经在 create_module 中设置好了
        pass

def install_file_index():
    """安装文件索引系统"""
    # 将查找器插入到导入系统的开头
    sys.meta_path.insert(0, FileIndexFinder())
    print("✅ 文件索引系统已安装")

def create_import_redirects():
    """为每个重定向创建 __init__.py 文件"""
    # 为 locomotion 子包创建 __init__.py
    locomotion_packages = [
        "locomotion",
        "locomotion/controllers",
        "locomotion/environments", 
        "locomotion/trainers",
        "locomotion/evaluators",
        "locomotion/utils",
        "scripts",
        "scripts/run",
        "scripts/test",
        "scripts/tools"
    ]
    
    for package in locomotion_packages:
        init_file = PROJECT_ROOT / package / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.write_text("# Auto-generated package\n")
            print(f"✅ 创建 {init_file}")

def list_file_mappings():
    """列出所有文件映射"""
    print("📁 文件映射表：")
    for old, new in FILE_MAPPINGS.items():
        print(f"  {old} -> {new}")

def verify_imports():
    """验证所有导入是否有效"""
    print("🔍 验证导入...")
    errors = []
    
    for old_name in FILE_MAPPINGS:
        try:
            module = importlib.import_module(old_name)
            print(f"✅ {old_name} -> {module.__file__}")
        except Exception as e:
            errors.append((old_name, str(e)))
    
    if errors:
        print("\n❌ 导入错误：")
        for old_name, error in errors:
            print(f"  {old_name}: {error}")
        return False
    
    print("\n✅ 所有导入验证通过")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="文件索引管理系统")
    parser.add_argument("--install", action="store_true", help="安装文件索引系统")
    parser.add_argument("--create-redirects", action="store_true", help="创建导入重定向")
    parser.add_argument("--list", action="store_true", help="列出文件映射")
    parser.add_argument("--verify", action="store_true", help="验证导入")
    
    args = parser.parse_args()
    
    if args.install:
        install_file_index()
    
    if args.create_redirects:
        create_import_redirects()
    
    if args.list:
        list_file_mappings()
    
    if args.verify:
        # 先安装索引系统
        install_file_index()
        verify_imports()