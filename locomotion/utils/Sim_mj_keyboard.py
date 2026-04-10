import threading
from pynput import keyboard
import math

class KeyboardCommander:
    """
    通用键盘控制器 (方向键版)
    功能：通过方向键控制线速度和偏航角
    特点：避免与 MuJoCo 常用快捷键冲突
    """
    def __init__(self, v_step=0.1, yaw_step_deg=10.0, verbose=True):
        self.target_vel = 0.0
        self.target_yaw = 0.0
        
        self.v_step = v_step
        self.yaw_step = yaw_step_deg * (math.pi / 180.0) # 弧度
        self.verbose = verbose
        
        self._lock = threading.Lock()
        
        # 启动监听
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        
        if self.verbose:
            self._print_instructions()

    def _print_instructions(self):
        
        print("\n" + "="*40)
        print(" [键盘控制器已更新] ")
        print(" ↑ (Up)    : 加速 (+Velocity)")
        print(" ↓ (Down)  : 减速 (-Velocity)")
        print(" ← (Left)  : 左转 (+Yaw Target)")
        print(" → (Right) : 右转 (-Yaw Target)")
        print(" Enter     : 急停 (速度归零)")
        print("="*40 + "\n")

    def _on_press(self, key):
        try:
            with self._lock:
                # 识别特殊按键
                if key == keyboard.Key.up:
                    self.target_vel += self.v_step
                elif key == keyboard.Key.down:
                    self.target_vel -= self.v_step
                elif key == keyboard.Key.left:
                    # 左转增加 Yaw 角度 (右手定则，Z轴朝上)
                    self.target_yaw += self.yaw_step
                elif key == keyboard.Key.right:
                    self.target_yaw -= self.yaw_step

                elif key == keyboard.Key.backspace:
                    self.reset_flag = True 
                    
                elif key == keyboard.Key.enter:
                    self.target_vel = 0.0                    
                    if self.verbose: print("\n-> [急停] 速度已重置")

            # 打印当前指令状态
            if self.verbose and key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
                print(f"\rCmd -> Vel: {self.target_vel:.2f} m/s | Yaw: {math.degrees(self.target_yaw):.1f} deg", end="")

        except Exception as e:
            pass

    def get_command(self):
        with self._lock:
            return self.target_vel, self.target_yaw
    
    def get_reset_flag(self):
        with self._lock:
            flag = getattr(self, 'reset_flag', False)
            self.reset_flag = False  # 读取后重置标志
            return flag

    def stop(self):
        self.listener.stop()