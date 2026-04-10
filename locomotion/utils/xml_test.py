import time
import math
import numpy as np
import mujoco
import mujoco.viewer
from transforms3d import euler

def check_sensors():
    print("\n" + "="*50)
    print("🚀 MuJoCo 传感器体检诊断系统启动")
    print("="*50 + "\n")

    # 1. 加载你的模型
    xml_path = "/home/huang/wheel_leg/wheel_legged_genesis_new/assets/description/scence_Sim.xml"
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ 模型加载成功！\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 获取关节 ID
    joint_names = ["L1_joint", "L2_joint", "L3_joint", "R1_joint", "R2_joint", "R3_joint"]
    joint_ids = {}
    print("--- 关节 ID 映射检查 ---")
    for name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ids[name] = j_id
        if j_id == -1:
            print(f"❌ 警告：未找到关节 {name}！")
        else:
            print(f"   {name}: ID = {j_id}")
    print("--------------------------\n")

    # 3. 开启被动渲染循环
    with mujoco.viewer.launch_passive(model, data) as viewer:
        model.opt.timestep = 0.002
        step = 0

        print("🚨 请在弹出的画面中，用鼠标拖拽机器人进行测试！")
        print("🚨 数据每 1 秒刷新一次...")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step += 1
            if step % 500 == 0:  # 每 1 秒打印一次 (500 * 0.002 = 1.0s)
                # ================= 数据读取 =================
                # 1. 提取四元数并转换为欧拉角
                quat = data.qpos[3:7]  # MuJoCo 默认为 [w, x, y, z]
                try:
                    rpy = euler.quat2euler(quat)
                    roll = rpy[0] * 180 / math.pi
                    pitch = rpy[1] * 180 / math.pi
                    yaw = rpy[2] * 180 / math.pi
                except Exception as e:
                    roll, pitch, yaw = 0, 0, 0
                    print(f"四元数转换错误: {e}")

                # 2. 提取全局线速度
                v_x, v_y, v_z = data.qvel[0], data.qvel[1], data.qvel[2]
                
                # 3. 模拟你的机体线速度计算逻辑
                yaw_rad = rpy[2]
                robot_x_vel = v_x * math.cos(yaw_rad) + v_y * math.sin(yaw_rad)

                # 4. 提取关节状态
                l_hip = data.qpos[joint_ids["L1_joint"]] * 180 / math.pi
                l_knee = data.qpos[joint_ids["L2_joint"]] * 180 / math.pi
                l_wheel_vel = data.qvel[joint_ids["L3_joint"]]
                
                r_hip = data.qpos[joint_ids["R1_joint"]] * 180 / math.pi
                r_knee = data.qpos[joint_ids["R2_joint"]] * 180 / math.pi
                r_wheel_vel = data.qvel[joint_ids["R3_joint"]]

                # ================= 打印诊断面板 =================
                print("\n" + "-"*40)
                print(f"⏱️ 仿真时间: {data.time:.2f} s")
                print("\n【1. IMU 姿态检查】(单位: 度)")
                print(f"   👉 Roll (侧倾):  {roll:+.2f}°")
                print(f"   👉 Pitch(俯仰):  {pitch:+.2f}°  <-- 前俯为正还是负？需牢记！")
                print(f"   👉 Yaw  (偏航):  {yaw:+.2f}°")

                print("\n【2. 速度投影检查】(单位: m/s)")
                print(f"   🌍 全局坐标系: Vx={v_x:+.2f}, Vy={v_y:+.2f}, Vz={v_z:+.2f}")
                print(f"   🤖 机体X轴(算出的): {robot_x_vel:+.2f} m/s <-- 往前推，它必须是正数！")

                print("\n【3. 腿部关节映射检查】(单位: 度)")
                print(f"   🦵 左腿: Hip={l_hip:+.1f}°, Knee={l_knee:+.1f}°")
                print(f"   🦵 右腿: Hip={r_hip:+.1f}°, Knee={r_knee:+.1f}°")
                
                print("\n【4. 轮子转速检查】(单位: rad/s)")
                print(f"   🛞 左轮速: {l_wheel_vel:+.2f}")
                print(f"   🛞 右轮速: {r_wheel_vel:+.2f} <-- 同向推车时，两轮符号是否一致？")
                print("-"*40)

            time.sleep(0.002)

if __name__ == "__main__":
    check_sensors()