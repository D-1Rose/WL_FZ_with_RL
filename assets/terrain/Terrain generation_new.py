import numpy as np
import cv2
import os

# 全局物理参数
horizontal_scale = 0.02   
vertical_scale = 0.001    
width_m = 2.0             
width_px = int(width_m / horizontal_scale) 
save_dir = "./assets/terrain/png"
os.makedirs(save_dir, exist_ok=True)

def generate_and_log_xml(profile_m, name, length_m, height_m):
    """保存图像并生成 MuJoCo XML 配置建议"""
    # 图像转换与保存
    profile_px = np.clip(profile_m / vertical_scale, 0, 255).astype(np.uint8)
    terrain_img = np.tile(profile_px, (width_px, 1))
    save_path = os.path.join(save_dir, f"{name}.png")
    cv2.imwrite(save_path, terrain_img)
    
    # 计算 MuJoCo size 参数: [半长, 半宽, 最大高度, 底座厚度]
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    
    print(f"\n✅ 地形生成成功: {name}")
    print(f"   物理总长: {length_m:.4f} m")
    print(f"   MuJoCo XML 配置参考:")
    print(f'   <hfield name="{name}" file="{os.path.abspath(save_path)}" ')
    print(f'           size="{half_length:.4f} {half_width:.4f} {height_m:.4f} 0.0001" />')

def generate_wave_only():
    """1. 纯波浪地形 (4个波)"""
    wave_len_m = 4.0
    wave_amp = 0.04  # 峰峰值 0.08m
    wave_period = 1.0
    
    length_px = int(wave_len_m / horizontal_scale)
    x = np.arange(length_px) * horizontal_scale
    profile_m = wave_amp * (1 - np.cos(2 * np.pi * x / wave_period))
    
    generate_and_log_xml(profile_m, "terrain_wave", wave_len_m, wave_amp * 2)

def generate_trapezoid_only(angle_deg, height, name):
    """2-4. 纯梯形斜坡 (无前后缓冲)"""
    ramp_len = height / np.tan(np.radians(angle_deg))
    flat_len = 1.0
    total_len_m = 2 * ramp_len + flat_len
    
    length_px = int(np.ceil(total_len_m / horizontal_scale))
    real_total_len = length_px * horizontal_scale
    x = np.arange(length_px) * horizontal_scale
    
    profile_m = np.zeros(length_px, dtype=np.float32)
    
    # 上坡
    m_up = (x >= 0) & (x < ramp_len)
    profile_m[m_up] = x[m_up] / ramp_len * height
    # 平台
    m_flat = (x >= ramp_len) & (x < ramp_len + flat_len)
    profile_m[m_flat] = height
    # 下坡
    m_down = (x >= ramp_len + flat_len)
    profile_m[m_down] = height - (x[m_down] - (ramp_len + flat_len)) / ramp_len * height
    
    generate_and_log_xml(np.clip(profile_m, 0, None), name, real_total_len, height)

if __name__ == "__main__":
    print("🚀 正在生成专项测试地形并计算 XML 参数...")
    
    # 生成波浪
    generate_wave_only()
    
    # 生成三个斜坡
    generate_trapezoid_only(10.0, 0.20, "terrain_slope_10")
    generate_trapezoid_only(15.0, 0.20, "terrain_slope_15")
    generate_trapezoid_only(25.0, 0.20, "terrain_slope_25")