import pygame
import numpy as np
from pathlib import Path

class control_gamepad:
    def __init__(self,command_cfg,command_scale=None):
        pygame.init()
        pygame.joystick.init()
        self.use_gamepad = True

        # 获取连接的游戏手柄数量
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("no gamepad,open keyboard window")
            self.use_gamepad = False
            screen_width = 300
            screen_height = 480
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            image_path1 = "../picture/keyboard_key.png" #sim2sim
            image_path2 = "picture/keyboard_key.png"
            image_center = (160, 240)
            pygame.display.set_caption("请用此窗口进行键盘控制(This use your keyboard)")
            image_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
            image_surface.fill((255, 255, 255, 0))  # 透明初始化
            try:
                if Path(image_path1).exists():
                    image_surface = pygame.image.load(image_path1)
                elif Path(image_path2).exists():
                    image_surface = pygame.image.load(image_path2)
                else:
                    print(f"无法加载图片: picture/keyboard_key.png")
            except pygame.error as e:
                print(f"无法加载图片: picture/keyboard_key.png")
                print(e)
                pygame.quit()
                exit()
            image_rect = image_surface.get_rect()
            image_rect.center = image_center
            self.screen.fill((255, 255, 255)) # 背景
            self.screen.blit(image_surface, image_rect)
            pygame.display.flip() # 更新屏幕显示 (一次性) 
        else:
            # 选择第一个手柄
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"link gamepad: {self.joystick.get_name()}")
        self.num_commands = command_cfg["num_commands"]
        self.command_cfg = command_cfg
        self.commands = np.zeros(self.num_commands)
        self.command_scale = command_scale
        if self.command_scale is None:
            self.command_scale = [1.0, 1.0, 3.14 ,1]
    
    def get_commands(self):

        """
        事件处理：监听游戏手柄或键盘的输入事件。
        输入映射：将物理输入（如按键、摇杆移动）转换为机器人的控制命令（如速度、转向角度）。
        命令限幅：确保输出的控制命令在安全范围内，避免物理引擎过载。

        键盘:
        监听键盘按键事件（按下和释放）。
        将按键操作映射为机器人的控制命令（如前进、后退、转向、调整身高）。
        处理按键释放时的命令归零，确保机器人在松开按键后停止相应动作。

        """

        pygame.event.pump()  # # 更新Pygame事件队列
        reset_flag = False   # # 初始化重置标志（默认为False）
        if self.use_gamepad:
            for event in pygame.event.get():  # # 遍历所有手柄事件
                if event.type == pygame.QUIT:  #  如果用户关闭窗口，则退出程序
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN:  # 手柄按钮按下事件
                    # print(f"按钮 {event.button} 被按下。")
                    match event.button:
                        case 0:
                            reset_flag=True
                # elif event.type == pygame.JOYBUTTONUP:
                #     print(f"按钮 {event.button} 被释放。")
                elif event.type == pygame.JOYAXISMOTION:   # 手柄摇杆移动事件
                    # print(f"轴 {event.axis},{event.value}")
                    match event.axis:
                        case 1: ## 左摇杆Y轴（LY轴） 前正后负
                            self.commands[0] = -event.value * self.command_scale[0]
                        case 2: # 右摇杆X轴（RX轴） 左正右负
                            self.commands[2] = -event.value * self.command_scale[2]
                        case 3: #lt 增加身高
                            self.commands[3] += (event.value+1) * self.command_scale[3]
                        case 4: #ry 减少身高
                            self.commands[3] -= (event.value+1) * self.command_scale[3]
        else:
            quiet_walking = 1.0
            for event in pygame.event.get():  # 获取事件队列中的所有事件
                if event.type == pygame.QUIT:  # 用户点击窗口关闭按钮
                    running = False
                elif event.type == pygame.KEYDOWN:  # 键盘按键按下事件
                    match event.key:
                        case pygame.K_w:
                            self.commands[0] += self.command_scale[0] * quiet_walking
                        case pygame.K_s:
                            self.commands[0] -= self.command_scale[0] * quiet_walking
                        case pygame.K_a:
                            self.commands[1] = self.command_scale[1] * quiet_walking
                        case pygame.K_d:
                            self.commands[1] = -self.command_scale[1] * quiet_walking
                        case pygame.K_q:
                            self.commands[2] = +self.command_scale[2] * quiet_walking
                        case pygame.K_e:
                            self.commands[2] = -self.command_scale[2] * quiet_walking
                        case pygame.K_SPACE:
                            self.commands[3] = +self.command_scale[3] * 100.0
                        case pygame.K_c:
                            self.commands[3] = -self.command_scale[3] * 100.0
                        case pygame.K_LCTRL:
                            self.commands[3] = -self.command_scale[3] * 100.0
                        case pygame.K_r:
                            reset_flag=True
                        case pygame.K_LSHIFT:
                            quiet_walking=0.2
                elif event.type == pygame.KEYUP:  # 键盘按键释放事件
                    match event.key:
                        # case pygame.K_w:
                        #     self.commands[0] = 0
                        # case pygame.K_s:
                        #     self.commands[0] = 0
                        case pygame.K_a:
                            self.commands[1] = 0
                        case pygame.K_d:
                            self.commands[1] = 0
                        case pygame.K_q:
                            self.commands[2] = 0
                        case pygame.K_e:
                            self.commands[2] = 0
        self.commands_clip()
        return self.commands,reset_flag
    
    def commands_clip(self):
        # lin_vel_x
        # if self.commands[0] < self.command_cfg["lin_vel_x_range"][0] * self.command_scale[0]:
        #     self.commands[0] = self.command_cfg["lin_vel_x_range"][0] * self.command_scale[0]
        # elif self.commands[0] > self.command_cfg["lin_vel_x_range"][1] * self.command_scale[0]:
        #     self.commands[0] = self.command_cfg["lin_vel_x_range"][1] * self.command_scale[0]

        #lin_vel_y
        if self.commands[1] < self.command_cfg["lin_vel_y_range"][0] * self.command_scale[1]:
            self.commands[1] = self.command_cfg["lin_vel_y_range"][0] * self.command_scale[1]
        elif self.commands[1] > self.command_cfg["lin_vel_y_range"][1] * self.command_scale[1]:
            self.commands[1] = self.command_cfg["lin_vel_y_range"][1] * self.command_scale[1]

        #ang_vel
        if self.commands[2] < self.command_cfg["ang_vel_range"][0] * self.command_scale[2]:
            self.commands[2] = self.command_cfg["ang_vel_range"][0] * self.command_scale[2]
        elif self.commands[2] > self.command_cfg["ang_vel_range"][1] * self.command_scale[2]:
            self.commands[2] = self.command_cfg["ang_vel_range"][1] * self.command_scale[2]

        #base_heigh
        if self.commands[3] < self.command_cfg["height_target_range"][0]:
            self.commands[3] = self.command_cfg["height_target_range"][0]
        elif self.commands[3] > self.command_cfg["height_target_range"][1]:
            self.commands[3] = self.command_cfg["height_target_range"][1]