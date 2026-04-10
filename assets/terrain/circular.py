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
    cv2.circle(slope, (slope_center,slope_center), r, color, thickness)
    
slope = cv2.blur(slope,(5,5))
print(f"horizontal_scale:{horizontal_scale} vertical_scale:{vertical_scale}")
print(f"terrain size:{(img_size*horizontal_scale*2,img_size*horizontal_scale)}")
print(f"World Respawn points:\n{(slope_center * horizontal_scale,slope_center * horizontal_scale,0.0)}")
# 保存图像
cv2.imwrite("./png/circular.png", slope)
