"""
环境模块 - 统一的环境接口
"""

# 环境类
__all__ = [
    # 将在后续步骤中从locomotion提取
]

# 版本信息
__version__ = "0.1.0"
__description__ = "轮腿机器人仿真环境"


def create_env(env_name="wheel_legged", **kwargs):
    """
    创建环境实例
    
    Args:
        env_name: 环境名称
        **kwargs: 环境参数
        
    Returns:
        环境实例
    """
    # 这里将实现从locomotion导入环境
    # 暂时返回None，后续实现
    print(f"创建环境: {env_name}")
    print(f"参数: {kwargs}")
    return None


if __name__ == "__main__":
    print(f"环境模块 v{__version__}")
    print(f"描述: {__description__}")