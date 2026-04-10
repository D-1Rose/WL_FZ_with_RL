"""
路径工具 - 解决硬编码路径问题
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    自动获取项目根目录
    
    Returns:
        Path: 项目根目录路径
    """
    # 方法1：从当前文件向上查找
    current_file = Path(__file__).resolve()
    
    # 向上查找3层（src/utils/..）
    root = current_file.parent.parent.parent
    
    # 验证是否是项目根目录（包含assets、locomotion等目录）
    if (root / "assets").exists() and (root / "locomotion").exists():
        return root
    
    # 方法2：尝试从环境变量获取
    env_root = os.environ.get("WHEEL_LEG_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    
    # 方法3：使用当前工作目录
    cwd = Path.cwd()
    if (cwd / "assets").exists() and (cwd / "locomotion").exists():
        return cwd
    
    raise FileNotFoundError(
        "无法确定项目根目录。请确保在项目目录中运行，"
        "或设置 WHEEL_LEG_ROOT 环境变量。"
    )


class ProjectPaths:
    """统一管理项目路径"""
    
    def __init__(self):
        self.root = get_project_root()
        
    @property
    def assets(self) -> Path:
        return self.root / "assets"
    
    @property
    def configs(self) -> Path:
        return self.root / "configs"
    
    @property
    def src(self) -> Path:
        return self.root / "src"
    
    @property
    def experiments(self) -> Path:
        return self.root / "experiments"
    
    @property
    def logs(self) -> Path:
        return self.root / "logs"
    
    @property
    def models(self) -> Path:
        return self.root / "models"
    
    def get_robot_path(self, robot_name: str = "wheel_leg", format: str = "urdf") -> Path:
        """
        获取机器人模型路径
        
        Args:
            robot_name: 机器人名称
            format: 模型格式 (urdf, mjcf, xml)
            
        Returns:
            Path: 机器人模型文件路径
        """
        if format == "urdf":
            return self.assets / "description" / "urdf" / f"{robot_name}.urdf"
        elif format == "mjcf":
            return self.assets / "mjcf" / "wheel_leg" / f"{robot_name}.xml"
        elif format == "xml":
            return self.assets / "description" / "urdf" / f"{robot_name}.xml"
        else:
            raise ValueError(f"不支持的模型格式: {format}")
    
    def get_terrain_path(self, terrain_name: str) -> Path:
        """
        获取地形文件路径
        
        Args:
            terrain_name: 地形名称
            
        Returns:
            Path: 地形文件路径
        """
        # 尝试多个可能的扩展名和位置
        possible_paths = [
            self.assets / "terrain" / "png" / f"{terrain_name}.png",
            self.assets / "terrain" / f"{terrain_name}.png",
            self.assets / "terrain" / "png" / f"{terrain_name}",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"找不到地形文件: {terrain_name}")
    
    def get_config_path(self, config_type: str, config_name: str) -> Path:
        """
        获取配置文件路径
        
        Args:
            config_type: 配置类型 (robots, terrains, experiments)
            config_name: 配置名称
            
        Returns:
            Path: 配置文件路径
        """
        path = self.configs / config_type / f"{config_name}.yaml"
        if not path.exists():
            # 如果不存在，尝试创建默认配置
            self._create_default_config(config_type, config_name)
        return path
    
    def _create_default_config(self, config_type: str, config_name: str):
        """创建默认配置文件"""
        config_dir = self.configs / config_type
        config_dir.mkdir(exist_ok=True, parents=True)
        
        # 根据类型创建不同的默认配置
        if config_type == "robots":
            default_content = f"""# 机器人配置: {config_name}
robot:
  name: "{config_name}"
  type: "urdf"
  file: "wheel_leg.urdf"
  init_pos: [0.0, 0.0, 0.3]
  init_quat: [1.0, 0.0, 0.0, 0.0]
"""
        elif config_type == "terrains":
            default_content = f"""# 地形配置: {config_name}
terrain:
  name: "{config_name}"
  file: "{config_name}.png"
  horizontal_scale: 0.02
  vertical_scale: 0.001
  type: "wave"
"""
        else:
            default_content = f"""# 实验配置: {config_name}
experiment:
  name: "{config_name}"
  description: "实验配置"
"""
        
        config_file = config_dir / f"{config_name}.yaml"
        config_file.write_text(default_content)
        print(f"📝 创建默认配置文件: {config_file}")


# 全局实例
paths = ProjectPaths()


if __name__ == "__main__":
    # 测试路径工具
    print("🔍 测试路径工具:")
    print(f"项目根目录: {paths.root}")
    print(f"资源目录: {paths.assets}")
    print(f"URDF路径: {paths.get_robot_path('wheel_leg', 'urdf')}")
    print(f"地形路径: {paths.get_terrain_path('terrain_wave')}")