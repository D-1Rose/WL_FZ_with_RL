import numpy as np
import cv2
import math

def generate_exact_ramp_hfield():
    # ================= 配置参数 =================
    slope_angle_deg = 35        # 坡度 35度
    ramp_height = 0.15          # 坡顶高度 150mm = 0.15m
    flat_top_length = 0.6       # 顶部平面长 600mm = 0.6m
    
    map_width_y = 1.0           # 地形宽度 1米 (Y方向，保持不变)
    resolution = 2000           # 分辨率设高一点 (每米2000像素)，保证边缘锐利
    
    # ================= 精确几何计算 =================
    
    # 1. 计算单侧斜坡的水平投影长度
    # L = H / tan(theta)
    slope_angle_rad = math.radians(slope_angle_deg)
    slope_run_length = ramp_height / math.tan(slope_angle_rad)
    
    # 2. 计算整个地形需要的精确总长 (无多余平地)
    # 总长 = 上坡长 + 顶部长 + 下坡长
    map_length_x = slope_run_length * 2 + flat_top_length
    
    print(f"--- 几何参数 ---")
    print(f"坡度: {slope_angle_deg}°")
    print(f"坡高: {ramp_height}m")
    print(f"单侧坡长: {slope_run_length:.5f}m")
    print(f"顶部平长: {flat_top_length:.5f}m")
    print(f"==> 地图总长 (X): {map_length_x:.5f}m")  # 这应该是 1.02844m 左右
    
    # ================= 生成图像 =================
    
    # 根据物理尺寸计算像素尺寸
    img_h = int(map_width_y * resolution)
    img_w = int(map_length_x * resolution)
    
    print(f"生成图像尺寸: {img_w} x {img_h} (像素)")
    
    # 创建高度场
    hfield = np.zeros((img_h, img_w), dtype=np.float32)
    
    # 关键点像素索引 (从0开始，填满整个宽度)
    start_x = 0
    top_start_x = int(slope_run_length * resolution)
    top_end_x = top_start_x + int(flat_top_length * resolution)
    end_x = img_w  # 最后一个像素
    
    # 生成剖面 (X轴)
    profile = np.zeros(img_w, dtype=np.float32)
    
    for x in range(img_w):
        # 1. 上坡段
        if x < top_start_x:
            # 高度 = (当前距离 / 坡长) * 最大高度
            dist = x / resolution
            h = dist * math.tan(slope_angle_rad)
            profile[x] = h
            
        # 2. 顶部平坦段
        elif top_start_x <= x < top_end_x:
            profile[x] = ramp_height
            
        # 3. 下坡段
        else:
            # 计算离右边缘还有多远
            dist_from_end = (end_x - x) / resolution
            h = dist_from_end * math.tan(slope_angle_rad)
            profile[x] = h
    
    # 修正：防止计算误差导致最高点略微超过或低于 ramp_height
    # 强制将顶部区域设为精确高度
    profile[top_start_x:top_end_x] = ramp_height
            
    # 广播到所有行
    hfield[:] = profile
    
    # ================= 输出 16-bit PNG =================
    
    # 归一化到 0-65535
    # 注意：这里我们让 65535 对应 ramp_height
    # 这样在 XML 里设置 Z-scale = ramp_height 即可
    normalized_hfield = (hfield / ramp_height) * 65535
    normalized_hfield = np.clip(normalized_hfield, 0, 65535).astype(np.uint16)
    
    filename = "ramp_35deg_exact.png"
    cv2.imwrite(filename, normalized_hfield)
    print(f"\n图片已保存: {filename}")
    
    # ================= 生成 XML 代码 =================
    print("\n" + "="*20 + " 请使用以下 XML 参数 " + "="*20)
    
    # MuJoCo size = [x半长, y半长, z高度, z厚度]
    half_x = map_length_x / 2
    half_y = map_width_y / 2
    
    print(f"")
    print(f'<hfield name="terrain" file="{filename}" size="{half_x:.4f} {half_y:.4f} {ramp_height} 0.1" />')
    print("="*60)

if __name__ == "__main__":
    generate_exact_ramp_hfield()