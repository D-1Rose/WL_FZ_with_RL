import cv2
import numpy as np
import math

img_size = 100
horizontal_scale = 0.1
vertical_scale = 0.001   #垂直缩放(单像素高度)
min_height = 0.01/0.001

#################### 连续坡路 ####################
slope = np.full((img_size, img_size), min_height, dtype=np.uint8)
slope_center = img_size//2
angle_degrees = 17
thickness = 1
max_height = math.tan(math.radians(angle_degrees))*(img_size*horizontal_scale) #现实高度
flag = 1 #1上坡 -1下坡 0维持
keep = 5
cnt = 0
color = min_height
for r in range(10,img_size):
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
    # cv2.circle(slope, (center,center), r, color, thickness)
    x1 = int(slope_center - r / 2)
    y1 = int(slope_center - r / 2)
    x2 = int(slope_center + r / 2)
    y2 = int(slope_center + r / 2)
    cv2.rectangle(slope, (x1, y1), (x2, y2), color, thickness)
    
#边缘入口
top_left = 0
color = 0
thickness = 1
for i in range(0,2):
    color +=i*5
    cv2.rectangle(slope, (top_left+i,top_left+i), (img_size-i,img_size-i), color, thickness)
    
#################### 崎岖路面 ####################
mean_noise_hight = 0.05 #噪声平均高度 m
rugged = np.random.randint(0, mean_noise_hight/vertical_scale, (img_size, img_size), dtype=np.uint8)

#平地位置
rugged_center = img_size//2
center_hight = 80
plane_size = 4
rugged[rugged_center-plane_size:rugged_center+plane_size, rugged_center-plane_size:rugged_center+plane_size] = center_hight

#边缘围墙
top_left = 0
color = 0
thickness = 1
for i in range(0,2):
    color +=i*5
    cv2.rectangle(slope, (top_left+i,top_left+i), (img_size-i,img_size-i), color, thickness)

img_concat_v = np.vstack((slope, rugged))

print(f"horizontal_scale:{horizontal_scale} vertical_scale:{vertical_scale}")
print(f"terrain size:{(img_size*horizontal_scale*2,img_size*horizontal_scale)}")
print(f"World Respawn points:\n{(slope_center * horizontal_scale,slope_center * horizontal_scale,0.0)},"
      f"\n{((rugged_center+img_size) * horizontal_scale,rugged_center * horizontal_scale,center_hight * vertical_scale)}")
# 保存图像
cv2.imwrite("./png/agent_train_gym.png", img_concat_v)
