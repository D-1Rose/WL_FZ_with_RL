import cv2
import numpy as np
import os

# 参数设置
stairs_width = 0.2
stairs_height = 0.2
horizontal_scale = 0.05  # 调整水平比例
vertical_scale = 0.02    # 调整垂直比例
min_height = 50          # 提高基础高度
img_size = 100          # 固定尺寸便于控制

# 创建地形矩阵
stairs = np.full((img_size, img_size), min_height, dtype=np.uint8)
stairs_center = img_size // 2
thickness = int(stairs_width / horizontal_scale)
color_step = 255 // (img_size // (thickness + 3))  # 计算颜色步长[9]

current_color = min_height
for r in range(10, img_size, thickness + 3):  # 增加起始偏移
    cv2.rectangle(stairs, 
                 (stairs_center-r, stairs_center-r),
                 (stairs_center+r, stairs_center+r),
                 int(current_color), thickness)
    current_color += color_step
    current_color = min(current_color, 255)  # 手动限幅

# 保存结果
os.makedirs("./png", exist_ok=True)
cv2.imwrite("./png/stairs.png", stairs)

print(f"当前颜色梯度步长: {color_step}")
print(f"最终颜色值: {current_color}")