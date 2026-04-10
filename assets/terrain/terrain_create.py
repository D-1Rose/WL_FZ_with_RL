import cv2
import numpy as np
import math
import os

# ================= 📝 参数配置区域 =================
SAVE_DIR = "./assets/terrain/png"

TASKS = {
    # [任务1] 10度坡 (自动计算尺寸)
    "slope_10deg_fit": {
        "h_scale": 0.05,         # 精度: 5cm (性能平衡点)
        "v_scale": 0.003,        # 高度精度: 3mm
        "params": {
            "angle": 10,         # 坡度
            "uphill_len": 3.0,   # 上坡段 (米)
            "platform_len": 2.0, # 平台段 (米)
            "downhill_len": 3.0, # 下坡段 (米)
            # 注：助跑(runway)已被移除，实现"起始即上坡"
        }
    },

    # [任务2] 15度坡
    "slope_15deg_fit": {
        "h_scale": 0.05,
        "v_scale": 0.004,        # 精度: 4mm
        "params": {
            "angle": 15,
            "uphill_len": 3.0,
            "platform_len": 2.0,
            "downhill_len": 3.0,
        }
    },

    # [任务3] 20度坡
    "slope_20deg_fit": {
        "h_scale": 0.05,
        "v_scale": 0.005,        # 精度: 5mm
        "params": {
            "angle": 20,
            "uphill_len": 3.0,
            "platform_len": 2.0,
            "downhill_len": 3.0,
        }
    }
}
# =================================================

class TerrainGeneratorFit:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"[INFO] Directory ready: {self.save_dir}")

    def generate(self, name, cfg):
        # 1. 提取参数
        h_scale = cfg["h_scale"]
        v_scale = cfg["v_scale"]
        params = cfg["params"]
        
        # 2. 【核心】自动计算刚刚好的尺寸 (Just Right!)
        # 总长度 = 上坡 + 平台 + 下坡
        total_len_m = params["uphill_len"] + params["platform_len"] + params["downhill_len"]
        img_size = int(math.ceil(total_len_m / h_scale))
        
        # 3. 几何计算
        angle_rad = math.radians(params["angle"])
        tan_angle = math.tan(angle_rad)
        
        # 计算各段的像素长度
        p_uphill = int(params["uphill_len"] / h_scale)
        p_plat   = int(params["platform_len"] / h_scale)
        p_down   = img_size - p_uphill - p_plat # 剩余的都给下坡，确保填满图片
        
        # 关键节点
        idx_uphill_end = p_uphill
        idx_plat_end   = idx_uphill_end + p_plat
        
        # 真实最高点
        max_h_real = params["uphill_len"] * tan_angle
        
        # 4. 生成数据
        row_data = np.zeros(img_size, dtype=np.uint8)
        max_pixel = 0
        
        for col in range(img_size):
            h_m = 0.0
            
            if col < idx_uphill_end:    
                # [阶段1] 直接上坡 (无助跑)
                dist = col * h_scale
                h_m = dist * tan_angle
                
            elif col < idx_plat_end:      
                # [阶段2] 平台
                h_m = max_h_real
                
            else:      
                # [阶段3] 下坡
                # dist是从平台结束开始算的
                dist = (col - idx_plat_end) * h_scale
                h_m = max_h_real - (dist * tan_angle)
                if h_m < 0: h_m = 0
            
            # 映射像素
            pixel_val = int(round(h_m / v_scale))
            
            if pixel_val > max_pixel: max_pixel = pixel_val
            if pixel_val > 255: pixel_val = 255
            if pixel_val < 0: pixel_val = 0
            
            row_data[col] = pixel_val

        # 5. 全宽铺满
        canvas = np.tile(row_data, (img_size, 1))
        
        # 6. 保存
        filename = f"{name}.png"
        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, canvas)
        
        # 7. 打印报告 (含自动生成的 Size)
        self._print_report(name, save_path, h_scale, v_scale, max_h_real, max_pixel, img_size)

    def _print_report(self, name, path, h_s, v_s, max_h, max_px, size):
        abs_path = os.path.abspath(path)
        print("-" * 60)
        print(f"[INFO] Generated: {name}")
        print(f"File Path: {abs_path}")
        print(f"Map Size: {size}x{size} pixels ({size*h_s:.2f}m x {size*h_s:.2f}m)")
        print(f"Max Height: {max_h:.4f} m | Max Pixel: {max_px}")
        
        if max_px > 255:
            rec_v = math.ceil((max_h / 255) * 1000) / 1000
            print(f"[WARNING] Overflow! Suggest setting 'v_scale' to {rec_v}")

        print("\n>>> Copy into FZ_RL_EVAL.py >>>")
        print(f'terrain_cfg["eval"] = "{name}"')
        print(f'terrain_cfg["horizontal_scale"] = {h_s}')
        print(f'terrain_cfg["vertical_scale"] = {v_s}')
        # 注意：这里不需要填 size，因为 eval 代码是读取图片后自动获取 size 的
        print("<<< End of block <<<")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    tool = TerrainGeneratorFit(SAVE_DIR)
    for task_name, config in TASKS.items():
        tool.generate(task_name, config)