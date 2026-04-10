import cv2
import numpy as np
import math

img_size = 104  # 图像尺寸
horizontal_scale = 0.1   # 水平缩放(单像素宽度) （像素到现实单位）
vertical_scale = 0.001   # 垂直缩放,将实际高度（单位：米）转换为像素值的系数 pixel_value = actual_height / vertical_scale
min_height = 0.01/0.001  # 最小高度(单位：像素)

# 生成一个100x100的随机噪声图像
mean_noise_hight = 0.05 # 噪声平均高度 m
agent_gym = np.random.randint(0, mean_noise_hight/vertical_scale, (img_size, img_size), dtype=np.uint8)

#边缘入口
top_left = 0
color = 0
thickness = 1
for i in range(0, 5):  # 在图像边缘绘制5层同心正方形，颜色从浅到深（模拟入口区域）
    color +=i*15
    cv2.rectangle(agent_gym, (top_left+i,top_left+i), (img_size-i,img_size-i), color, thickness)

#滑梯 2m宽 20像素
slide_thickness = 2.0
angle_degrees = 13.5
max_height = math.tan(math.radians(angle_degrees))*(img_size*horizontal_scale) #现实高度 

flag = 1 #1上坡 -1下坡 0维持
keep = 3
cnt = 0
color = min_height  # (10)
cv2.line(agent_gym, (0,0), (0,int(slide_thickness/horizontal_scale)), min_height, 1)  # # 绘制线条（图像，起点，终点，颜色，线宽）
for x in range(1,img_size):
    match flag:
        case 1:
            color += (1/img_size)*max_height / vertical_scale
        case -1:
            color -= (1/img_size)*max_height / vertical_scale
        case 0:
            cnt += 1
    if cnt>keep:
        if color==255:
            flag=-1
        elif color==min_height:
            flag=1
        cnt=0
    else:
        if color>255:
            color = 255
            flag=0
        elif color < 0:
            color=min_height
            flag=0
    cv2.line(agent_gym, (x,0), (x,int(slide_thickness/horizontal_scale)), color, 1)
    
print(f"horizontal_scale:{horizontal_scale} vertical_scale:{vertical_scale}")
print(f"terrain size:{img_size*horizontal_scale} m")
# 保存图像
import os

# ... (前面的代码保持不变) ...

print(f"horizontal_scale:{horizontal_scale} vertical_scale:{vertical_scale}")
print(f"terrain size:{img_size*horizontal_scale} m")

# --- 修改开始 ---
save_dir = "./assets/terrain/png"  # 或者修改为你项目里实际读取的路径，比如 "assets/terrain/png"
save_path = os.path.join(save_dir, "new.png")

# 1. 自动创建文件夹（如果不存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"已创建目录: {save_dir}")

# 2. 保存图像并检查结果
success = cv2.imwrite(save_path, agent_gym)

if success:
    # 获取绝对路径，方便你直接复制去查看
    abs_path = os.path.abspath(save_path)
    print(f"\n✅ 图片生成成功！")
    print(f"保存位置: {abs_path}")
else:
    print(f"\n❌ 图片保存失败！请检查路径或权限。")
# --- 修改结束 ---
