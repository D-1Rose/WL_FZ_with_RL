import numpy as np
import cv2
import os

# ==========================================
# 全局物理参数 (必须与你的 XML/Controller 匹配)
# ==========================================
horizontal_scale = 0.02   
vertical_scale = 0.001    
width_m = 2.0             
width_px = int(width_m / horizontal_scale) 
save_dir = "./assets/terrain/png"
os.makedirs(save_dir, exist_ok=True)

def generate_and_log_xml(profile_m, name, length_m, height_m):
    """保存图像并输出 MuJoCo XML 配置建议"""
    # 转换为灰度图像素 (0-255)
    profile_px = np.clip(profile_m / vertical_scale, 0, 255).astype(np.uint8)
    # 将 1D 轮廓平铺成 2D 地图
    terrain_img = np.tile(profile_px, (width_px, 1))
    
    save_path = os.path.join(save_dir, f"{name}.png")
    cv2.imwrite(save_path, terrain_img)
    
    # MuJoCo size: [半长, 半宽, 最大高度, 底座厚度]
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    
    print(f"\n✅ 地形生成成功: {name}")
    print(f"   物理总长: {length_m:.4f} m | 最大高度: {height_m:.4f} m")
    print(f"   [MuJoCo XML 属性参考]:")
    print(f'   <hfield name="{name}" file="{os.path.abspath(save_path)}" ')
    print(f'           size="{half_length:.4f} {half_width:.4f} {height_m:.4f} 0.0001" />')

def generate_dense_wave():
    """1. 高频密波浪 (专门测试谐振抑制)"""
    wave_len_m = 6.0
    wave_amp = 0.03    # 峰峰值 0.06m
    wave_period = 0.6  # 0.6m 一个周期
    
    length_px = int(wave_len_m / horizontal_scale)
    x = np.arange(length_px) * horizontal_scale
    # 构造正弦波，且让起点高度为 0
    profile_m = wave_amp * (1 - np.cos(2 * np.pi * x / wave_period))
    
    generate_and_log_xml(profile_m, "terrain_dense_wave", wave_len_m, wave_amp * 2)

def generate_short_bump():
    """2. 极限短平台 (专门测试瞬态脉冲抗扰)"""
    total_len_m = 4.0
    length_px = int(total_len_m / horizontal_scale)
    x = np.arange(length_px) * horizontal_scale
    profile_m = np.zeros(length_px, dtype=np.float32)
    
    height = 0.08      # 8cm 高度
    ramp_len = 0.15    # 极短上坡 (15cm 冲上 8cm，坡度积极)
    flat_len = 0.4     # 平台长度
    start_x = 1.0      # 起步平地
    
    # 上坡
    m_up = (x >= start_x) & (x < start_x + ramp_len)
    profile_m[m_up] = (x[m_up] - start_x) / ramp_len * height
    # 平台
    m_flat = (x >= start_x + ramp_len) & (x < start_x + ramp_len + flat_len)
    profile_m[m_flat] = height
    # 下坡
    m_down = (x >= start_x + ramp_len + flat_len) & (x < start_x + 2*ramp_len + flat_len)
    profile_m[m_down] = height - (x[m_down] - (start_x + ramp_len + flat_len)) / ramp_len * height
    
    generate_and_log_xml(np.clip(profile_m, 0, None), "terrain_short_bump", total_len_m, height)

def generate_rough_terrain():
    """3. 随机崎岖地形 (测试不确定性自适应)"""
    total_len_m = 5.0
    length_px = int(total_len_m / horizontal_scale)
    x = np.arange(length_px) * horizontal_scale # 修复了之前的定义缺失
    height_max = 0.05
    
    # 随机噪声生成
    np.random.seed(42)
    noise = np.random.uniform(0, height_max, length_px)
    
    # 滤波平滑，防止物理引擎因三角形过尖锐而穿模
    kernel_size = int(0.12 / horizontal_scale) 
    kernel = np.ones(kernel_size) / kernel_size
    profile_m = np.convolve(noise, kernel, mode='same')
    
    # 前 1 米强制平地
    profile_m[x < 1.0] = 0.0
    
    generate_and_log_xml(profile_m, "terrain_rough", total_len_m, height_max)

if __name__ == "__main__":
    print("🚀 启动针对二型模糊控制器的专项地形生成...")
    generate_dense_wave()
    generate_short_bump()
    generate_rough_terrain()
    print("\n💡 提示: 请将生成的 PNG 图片放入 assets/terrain/png 目录，并更新 XML 配置。")