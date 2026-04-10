"""
轮腿机器人项目 - 统一导入接口
"""

from .utils.path_utils import paths, ProjectPaths, get_project_root
from .utils.config_loader import (
    load_yaml, save_yaml, load_config, merge_configs,
    load_experiment_config, create_default_configs
)

# 版本信息
__version__ = "0.2.0"
__author__ = "轮腿机器人实验组"
__description__ = "基于Genesis的轮腿机器人强化学习平台"

# 导出常用工具
__all__ = [
    # 路径工具
    "paths",
    "ProjectPaths", 
    "get_project_root",
    
    # 配置工具
    "load_yaml",
    "save_yaml", 
    "load_config",
    "merge_configs",
    "load_experiment_config",
    "create_default_configs",
    
    # 元数据
    "__version__",
    "__author__",
    "__description__",
]


def setup_project():
    """
    设置项目环境
    
    Returns:
        dict: 项目配置信息
    """
    import sys
    from pathlib import Path
    
    # 添加项目根目录到Python路径
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 创建必要的目录
    for dir_name in ["logs", "models", "experiments/temp"]:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True, parents=True)
    
    print(f"✅ 项目设置完成")
    print(f"   根目录: {project_root}")
    print(f"   版本: {__version__}")
    
    return {
        "root": project_root,
        "version": __version__,
        "paths": paths,
    }


if __name__ == "__main__":
    # 测试导入
    info = setup_project()
    print(f"\n📦 可用导入:")
    for item in __all__:
        print(f"  • {item}")