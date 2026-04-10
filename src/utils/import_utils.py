"""
导入工具 - 解决交叉引用和模块导入问题
集成了文件索引系统，支持文件重定向
"""

import importlib
import sys
from pathlib import Path
from typing import Any, Optional
from .path_utils import get_project_root
import warnings


class ImportManager:
    """统一管理模块导入，解决交叉引用问题"""
    
    def __init__(self):
        self.project_root = get_project_root()
        self._modules_cache = {}
        
    def import_from_locomotion(self, module_name: str, class_name: Optional[str] = None) -> Any:
        """
        从locomotion目录导入模块或类
        支持新旧模块路径
        
        Args:
            module_name: 模块名（不含.py）
            class_name: 类名（可选）
            
        Returns:
            模块或类
        """
        cache_key = f"locomotion.{module_name}.{class_name}"
        
        if cache_key in self._modules_cache:
            return self._modules_cache[cache_key]
        
        # 尝试多种导入路径
        import_paths = [
            f"locomotion.{module_name}",  # 旧路径
            f"locomotion.controllers.{module_name}",  # 新路径 - 控制器
            f"locomotion.environments.{module_name}",  # 新路径 - 环境
            f"locomotion.trainers.{module_name}",  # 新路径 - 训练器
            f"locomotion.evaluators.{module_name}",  # 新路径 - 评估器
            f"locomotion.utils.{module_name}",  # 新路径 - 工具
        ]
        
        last_error = None
        
        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path)
                
                if class_name:
                    # 导入特定类
                    result = getattr(module, class_name)
                else:
                    # 导入整个模块
                    result = module
                
                # 缓存结果
                self._modules_cache[cache_key] = result
                
                # 如果使用了重定向路径，发出警告
                if import_path != f"locomotion.{module_name}":
                    warnings.warn(
                        f"模块 '{module_name}' 已移动到新位置: {import_path}\n"
                        f"请更新代码使用新导入路径。",
                        DeprecationWarning,
                        stacklevel=2
                    )
                
                return result
                
            except ImportError as e:
                last_error = e
                continue
            except AttributeError as e:
                # 模块存在但类不存在
                last_error = e
                continue
        
        # 所有导入尝试都失败
        print(f"❌ 导入失败: locomotion.{module_name}")
        print(f"   尝试的路径: {import_paths}")
        print(f"   最后错误: {last_error}")
        raise ImportError(f"无法导入 locomotion.{module_name}: {last_error}")
    
    def import_from_src(self, module_path: str, class_name: Optional[str] = None) -> Any:
        """
        从src目录导入模块或类
        
        Args:
            module_path: 模块路径（如 "core.controllers"）
            class_name: 类名（可选）
            
        Returns:
            模块或类
        """
        cache_key = f"src.{module_path}.{class_name}"
        
        if cache_key in self._modules_cache:
            return self._modules_cache[cache_key]
        
        try:
            # 从src导入
            full_path = f"src.{module_path}"
            module = importlib.import_module(full_path)
            
            if class_name:
                result = getattr(module, class_name)
            else:
                result = module
            
            self._modules_cache[cache_key] = result
            return result
            
        except ImportError as e:
            print(f"❌ 导入失败: src.{module_path}")
            print(f"   错误: {e}")
            raise
    
    def get_controller(self, controller_type: str = "fuzzy_pid"):
        """
        获取控制器实例
        
        Args:
            controller_type: 控制器类型
            
        Returns:
            控制器类
        """
        controller_map = {
            "fuzzy_pid": ("FZ_PID_c", "BatchedLQRController"),
            "vmc": ("FZ_PID_c", "BatchedVMC"),
            "pid": ("fuzzy_control", "PIDController"),
            "controller": ("Controller", "Controller"),
            "controller_text": ("controller_text", "ControllerText"),
            "offset_mj": ("Offset_mj_controller", "OffsetMjController"),
            "pitch": ("Pitch_controller", "PitchController"),
        }
        
        if controller_type not in controller_map:
            raise ValueError(f"未知控制器类型: {controller_type}")
        
        module_name, class_name = controller_map[controller_type]
        return self.import_from_locomotion(module_name, class_name)
    
    def get_env(self, env_type: str = "wheel_legged"):
        """
        获取环境类
        
        Args:
            env_type: 环境类型
            
        Returns:
            环境类
        """
        env_map = {
            "wheel_legged": ("FZ_PID_env_c", "WheelLeggedEnv"),
            "pitch": ("Pitch_env", "PitchEnv"),
            "mj": ("mj", "MjEnv"),
            "fz_rl": ("FZ_RL_ENV", "FZRLEnv"),
        }
        
        if env_type not in env_map:
            raise ValueError(f"未知环境类型: {env_type}")
        
        module_name, class_name = env_map[env_type]
        return self.import_from_locomotion(module_name, class_name)
    
    def clear_cache(self):
        """清空导入缓存"""
        self._modules_cache.clear()
        print("✅ 导入缓存已清空")


# 全局实例
import_manager = ImportManager()


# 便捷函数
def get_controller(controller_type: str = "fuzzy_pid"):
    """获取控制器（便捷函数）"""
    return import_manager.get_controller(controller_type)


def get_env(env_type: str = "wheel_legged"):
    """获取环境（便捷函数）"""
    return import_manager.get_env(env_type)


def import_module(module_path: str, from_src: bool = False):
    """
    导入模块
    
    Args:
        module_path: 模块路径
        from_src: 是否从src导入
        
    Returns:
        模块
    """
    if from_src:
        return import_manager.import_from_src(module_path)
    else:
        # 尝试多种导入方式
        try:
            # 首先尝试直接导入
            return importlib.import_module(module_path)
        except ImportError:
            # 如果失败，使用导入管理器
            if module_path.startswith("locomotion."):
                module_name = module_path.split(".")[-1]
                return import_manager.import_from_locomotion(module_name)
            else:
                raise


if __name__ == "__main__":
    # 测试导入工具
    print("🔍 测试导入工具...")
    
    # 测试导入控制器
    try:
        FuzzyPIDController = get_controller("fuzzy_pid")
        print(f"✅ 导入控制器: {FuzzyPIDController.__name__}")
    except Exception as e:
        print(f"❌ 导入控制器失败: {e}")
    
    # 测试导入环境
    try:
        WheelLeggedEnv = get_env("wheel_legged")
        print(f"✅ 导入环境: {WheelLeggedEnv.__name__}")
    except Exception as e:
        print(f"❌ 导入环境失败: {e}")
    
    print("\n✅ 导入工具测试完成")