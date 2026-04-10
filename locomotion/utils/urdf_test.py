import rsl_rl
print("🔥 当前使用的 rsl_rl 路径:", rsl_rl.__file__)
import genesis as gs
def main():
    gs.init(backend=gs.gpu)

    import matplotlib.pyplot as plt
    import time
    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options = gs.options.SimOptions(
            dt = 0.002,
        ),
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (0, -3.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 240,
        ),
        show_viewer = True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )    
    WL = scene.add_entity(
        gs.morphs.URDF(
            file  = '/home/huang/下载/wheel_leg/src/description/meshes/wheel_leg.urdf',
            pos   = (0, 0, 0.40),
            quat = (1.0, 0.0, 0.0, 0.0),
        ),
    )
    scene.build()





    while True:
    # for i in range(30):
        # 更新传感器数据
        jnt_names = [
            'L1_joint',
            'L2_joint',
            'R1_joint',
            'R2_joint',
            'L3_joint',
            'R3_joint',
        ]
        dofs_idx = [WL.get_joint(name).dof_idx_local for name in jnt_names]

        print(f"dofs_idx:{dofs_idx}")
        print(f"jnt_names:{jnt_names}")
        

        

        scene.step() 


        # print(lqr_controller.pitch,lqr_controller.pitch_dot)
        # now = time.time()
        # if now - last_print_time >= PRINT_INTERVAL_SEC:
        #     print(f"force: {lqr_controller.left_wheel_effort}, {lqr_controller.right_wheel_effort}")
        #     last_print_time = now
        # else :
        #     pass
                # print(f"pitch: {lqr_controller.pitch}, pitch_dot: {lqr_controller.pitch_dot}")
            # print(f"force: {lqr_controller.left_wheel_effort}, {lqr_controller.right_wheel_effort}")
        # while not rospy.is_shutdown():
            # 将膝髋关节的位置设置为0
            # b.command_pub_L1.publish(0)
            # b.command_pub_L2.publish(0)
            # b.command_pub_R1.publish(0)
            # b.command_pub_R2.publish(0)
        # a.vmc()
        # b.balance()
        # print(b.pitch,b.pitch_dot,b.yaw,b.yaw_dot,b.roll,b.roll_dot,b.robot_x_position,b.robot_x_velocity,-b.pitch_com,b.velocity_d)
        # b.rate.sleep()

if __name__ == '__main__':
    main()