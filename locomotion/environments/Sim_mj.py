import math
from transforms3d import euler
import numpy as np
import time
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use('TkAgg')  # Use a standalone window after MuJoCo viewer closes
import matplotlib.pyplot as plt

# --- PID 工具类 ---
class PIDController:
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        if self.output_limit is not None:
            output = max(min(output, self.output_limit), -self.output_limit)
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


# --- LQR Controller ---
class LQR_Controller:
    def __init__(self, model):
        self.model = model
        self.dt = 0.002
        self.base_quat = [1, 0, 0, 0]
        self.left_wheel_velocity = 0.0
        self.left_wheel_position = 0.0
        self.right_wheel_velocity = 0.0
        self.right_wheel_position = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.pitch_dot = 0.0
        self.yaw = 0.0
        self.yaw_d = 0.0
        self.yaw_dot = 0.0
        self.theta_r1 = 0; self.theta_r2 = 0; self.theta_r3 = 0; self.theta_rb = 0
        self.theta_l1 = 0; self.theta_l2 = 0; self.theta_l3 = 0; self.theta_lb = 0
        self.robot_x_velocity = 0
        self.robot_x_position = 0.0
        self.velocity_d = 0.0
        self.theta_pitch = 0.0
        self.pitch_com = 0.0

        self.command_yaw      = PIDController(kp=1,   ki=0.0,   kd=0.05)
        self.command_pitch    = PIDController(kp=25,  ki=0.0,   kd=1)
        self.command_velocity = PIDController(kp=0.3, ki=0.000, kd=0)

        _jmap = {
            "left_hip":    "L1_joint",
            "left_knee":   "L2_joint",
            "left_wheel":  "L3_joint",
            "right_hip":   "R1_joint",
            "right_knee":  "R2_joint",
            "right_wheel": "R3_joint",
        }
        self.joint_ids  = {}
        self.joint_qpos = {}
        self.joint_dof  = {}
        for key, jname in _jmap.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self.joint_ids[key]  = jid
            self.joint_qpos[key] = model.jnt_qposadr[jid]
            self.joint_dof[key]  = model.jnt_dofadr[jid]

        self.actuator_ids = {
            "left_hip":    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L1_joint_ctrl"),
            "left_knee":   mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L2_joint_ctrl"),
            "left_wheel":  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L3_joint_ctrl"),
            "right_hip":   mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R1_joint_ctrl"),
            "right_knee":  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R2_joint_ctrl"),
            "right_wheel": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R3_joint_ctrl"),
        }

        self.fuzzy_params = {
            "gamma_d_g": 75 * math.pi / 180,
            "eth": 0.05,
            "a": 0.05,
            "k_out": 1.0
        }

    def update_joint_states(self, data):
        self.left_wheel_position  = data.qpos[self.joint_qpos["left_wheel"]]
        self.right_wheel_position = data.qpos[self.joint_qpos["right_wheel"]]
        self.left_hip_position    = data.qpos[self.joint_qpos["left_hip"]]
        self.right_hip_position   = data.qpos[self.joint_qpos["right_hip"]]
        self.left_knee_position   = data.qpos[self.joint_qpos["left_knee"]]
        self.right_knee_position  = data.qpos[self.joint_qpos["right_knee"]]

        self.left_wheel_velocity  = data.qvel[self.joint_dof["left_wheel"]]
        self.right_wheel_velocity = data.qvel[self.joint_dof["right_wheel"]]
        self.left_knee_velocity   = data.qvel[self.joint_dof["left_knee"]]
        self.right_knee_velocity  = data.qvel[self.joint_dof["right_knee"]]
        self.left_hip_velocity    = data.qvel[self.joint_dof["left_hip"]]
        self.right_hip_velocity   = data.qvel[self.joint_dof["right_hip"]]

        self.theta_r1 = self.right_wheel_position
        self.theta_r2 = -1*(95/180 * math.pi - self.right_knee_position)
        self.theta_r3 = self.right_hip_position - 20/180 * math.pi
        self.theta_rb = -1*self.theta_r2 - self.theta_r3 - math.pi/2 + self.pitch
        self.theta_r1_dot = self.right_wheel_velocity
        self.theta_r2_dot = self.right_knee_velocity
        self.theta_r3_dot = self.right_hip_velocity
        self.theta_rb_dot = -self.theta_r2_dot - self.theta_r3_dot + self.pitch_dot

        self.theta_l1 = -self.left_wheel_position
        self.theta_l2 = -1*(95/180 * math.pi + self.left_knee_position)
        self.theta_l3 = -self.left_hip_position - 20/180 * math.pi
        self.theta_lb = -1*self.theta_l2 - self.theta_l3 - math.pi/2 + self.pitch
        self.theta_l1_dot = -self.left_wheel_velocity
        self.theta_l2_dot = -self.left_knee_velocity
        self.theta_l3_dot = -self.left_hip_velocity
        self.theta_lb_dot = -self.theta_l2_dot - self.theta_l3_dot + self.pitch_dot

        v_global_x = data.qvel[0]
        v_global_y = data.qvel[1]
        self.robot_x_velocity = v_global_x * math.cos(self.yaw) + v_global_y * math.sin(self.yaw)
        self.robot_x_position = (self.left_wheel_position - self.right_wheel_position) * 0.1 / 2.0

    def update_imu_data(self, data):
        quat = data.qpos[3:7]
        rpy  = euler.quat2euler(quat, axes='rzyx')
        self.yaw   = rpy[0]
        self.pitch = rpy[1]
        self.roll  = rpy[2]
        wx = data.qvel[3]; wy = data.qvel[4]; wz = data.qvel[5]
        cos_r, sin_r = math.cos(self.roll), math.sin(self.roll)
        cos_p = math.cos(self.pitch)
        tan_p = math.tan(self.pitch)
        if abs(cos_p) > 1e-4:
            self.roll_dot  = wx + sin_r*tan_p*wy + cos_r*tan_p*wz
            self.pitch_dot =           cos_r*wy  -        sin_r*wz
            self.yaw_dot   = (sin_r/cos_p)*wy    + (cos_r/cos_p)*wz
        else:
            self.roll_dot  = wx
            self.pitch_dot = wy
            self.yaw_dot   = wz

    def get_command_pitch(self, data):
        L, M, self.pitch_com, roll_com, Jz, Jy = self.com(
            self.right_hip_position, self.right_knee_position,
            self.left_hip_position,  self.left_knee_position, self.roll
        )
        pitch_d     = -self.pitch_com
        theta_pitch = pitch_d - self.pitch
        pitch_ref_from_vel = self.get_pitch_ref_from_velocity(data)
        self.theta_pitch = theta_pitch + pitch_ref_from_vel
        theta_pitch_pass = self.ST_SIT2_FLC_FM(self.theta_pitch)
        theta_pitch = self.theta_pitch
        command = self.command_pitch.compute(error=theta_pitch_pass, dt=self.model.opt.timestep)
        return command

    def get_pitch_ref_from_velocity(self, data):
        theta_velocity = self.velocity_d - self.robot_x_velocity
        pitch_offset = self.command_velocity.compute(error=theta_velocity, dt=self.model.opt.timestep)
        pitch_offset = np.clip(pitch_offset, -30.0*math.pi/180, 30.0*math.pi/180)
        return pitch_offset

    def get_command_yaw(self, data):
        raw_error = self.yaw_d - self.yaw
        theta_yaw = (raw_error + np.pi) % (2*np.pi) - np.pi
        output = self.command_yaw.compute(error=theta_yaw, dt=self.model.opt.timestep)
        output = np.clip(output, -3, 3)
        return output

    def com(self, xita_r1, xita_r2, xita_l1, xita_l2, roll):
        xita_r1 = xita_r1 - 20*math.pi/180
        xita_r2 = xita_r2 - 95*math.pi/180
        xita_l1 = -xita_l1 - 20*math.pi/180
        xita_l2 = -xita_l2 - 95*math.pi/180
        lr1 = 150*1e-3; lr2 = 250*1e-3; ll1 = lr1; ll2 = lr2; dlr = 286*1e-3
        dr1 = np.array([0.075, 0.0,  0.010]); dr2 = np.array([0.125, 0.0,  0.010])
        dl1 = np.array([0.075, 0.0, -0.010]); dl2 = np.array([0.125, 0.0, -0.010])
        db  = np.array([0, 0, 0])
        mr1 = 1.30110728; mr2 = 2.00310728; ml1 = 1.30110728; ml2 = 2.00310728; mb = 4.63282560
        cr1 = np.cos(xita_r1); sr1 = np.sin(xita_r1)
        cl1 = np.cos(xita_l1); sl1 = np.sin(xita_l1)
        cr1r2 = np.cos(xita_r1+xita_r2); sr1r2 = np.sin(xita_r1+xita_r2)
        cl1l2 = np.cos(xita_l1+xita_l2); sl1l2 = np.sin(xita_l1+xita_l2)
        Xr1 = np.array([cr1*dr1[0]-sr1*dr1[1], sr1*dr1[0]+cr1*dr1[1], dr1[2]+dlr/2])
        Xr2 = np.array([cr1r2*dr2[0]-sr1r2*dr2[1]+lr1*cr1, sr1r2*dr2[0]+cr1r2*dr2[1]+lr1*sr1, dr2[2]+dlr/2])
        Xl1 = np.array([cl1*dl1[0]-sl1*dl1[1], sl1*dl1[0]+cl1*dl1[1], dl1[2]-dlr/2])
        Xl2 = np.array([cl1l2*dl2[0]-sl1l2*dl2[1]+ll1*cl1, sl1l2*dl2[0]+cl1l2*dl2[1]+ll1*sl1, dl2[2]-dlr/2])
        Xb  = np.array([0, 0, 0])
        T_80_3 = np.array([
            lr2*np.cos(xita_r1+xita_r2)/2+lr2*np.cos(xita_l1+xita_l2)/2+lr1*np.cos(xita_l1)/2+lr1*np.cos(xita_r1)/2,
            lr2*np.sin(xita_l1+xita_l2)/2+lr2*np.sin(xita_r1+xita_r2)/2+lr1*np.sin(xita_l1)/2+lr1*np.sin(xita_r1)/2,
            0])
        Xbody = (mr1*Xr1+mr2*Xr2+ml1*Xl1+ml2*Xl2+mb*Xb)/(mr1+mr2+ml1+ml2+mb)
        X8_re = Xbody - T_80_3
        Rx_roll = np.array([[1,0,0],[0,np.cos(-roll),-np.sin(-roll)],[0,np.sin(-roll),np.cos(-roll)]])
        X8_re = np.dot(Rx_roll, X8_re)
        phi = np.arctan(X8_re[0]/X8_re[1])
        pitch_com = phi; roll_com = np.arctan(X8_re[2]/X8_re[1])
        L = np.sqrt(X8_re[0]**2+X8_re[1]**2)
        rr1 = np.sqrt((Xr1[0]-T_80_3[0])**2+(Xr1[1]-T_80_3[1])**2)
        rr2 = np.sqrt((Xr2[0]-T_80_3[0])**2+(Xr2[1]-T_80_3[1])**2)
        rl1 = np.sqrt((Xl1[0]-T_80_3[0])**2+(Xl1[1]-T_80_3[1])**2)
        rl2 = np.sqrt((Xl2[0]-T_80_3[0])**2+(Xl2[1]-T_80_3[1])**2)
        rb  = np.sqrt((Xb[0] -T_80_3[0])**2+(Xb[1] -T_80_3[1])**2)
        m = (mr2*rr2**2+ml2*rl2**2+mr1*rr1**2+ml1*rl1**2+mb*rb**2)/(L**2)
        J_z = m*L**2
        rr1_r = math.sqrt((Xr1[0]-T_80_3[0])**2+(Xr1[2]-T_80_3[2])**2)
        rr2_r = math.sqrt((Xr2[0]-T_80_3[0])**2+(Xr2[2]-T_80_3[2])**2)
        rl1_r = math.sqrt((Xl1[0]-T_80_3[0])**2+(Xl1[2]-T_80_3[2])**2)
        rl2_r = math.sqrt((Xl2[0]-T_80_3[0])**2+(Xl2[2]-T_80_3[2])**2)
        rb_r  = math.sqrt((Xb[0] -T_80_3[0])**2+(Xb[2] -T_80_3[2])**2)
        J_y = mr2*rr2_r**2+ml2*rl2_r**2+mr1*rr1_r**2+ml1*rl1_r**2+mb*rb_r**2
        return L, m, pitch_com, roll_com, J_z, J_y

    def ST_SIT2_FLC_FM(self, error_pass):
        DOMAIN_DEG = 15
        error   = error_pass/math.pi*180.0/DOMAIN_DEG
        MAX_RAD = DOMAIN_DEG/180.0*math.pi
        gamma_d_g = self.fuzzy_params["gamma_d_g"]
        eth = self.fuzzy_params["eth"]; a = self.fuzzy_params["a"]
        cos_gamma = np.cos(gamma_d_g)
        eth = np.minimum(eth, cos_gamma-0.001)
        ld_g = (cos_gamma-eth)/cos_gamma
        B0_g,B1_g,B2_g = 0.0,ld_g*np.sin(gamma_d_g),1.0
        C0_g,C1_g,C2_g = 0.0,ld_g*np.cos(gamma_d_g),1.0
        K0_g = (B1_g-B0_g)/(C1_g-C0_g); K1_g = (B2_g-B1_g)/(C2_g-C1_g)
        N0_g = (B0_g*C1_g-B1_g*C0_g)/(C1_g-C0_g); N1_g = (B1_g*C2_g-B2_g*C1_g)/(C2_g-C1_g)
        ld_max,ld_min = 0.75,0.1
        if error <= -C2_g:              gamma_g = B2_g*70.0+10.0
        elif -C2_g < error < -C1_g:    gamma_g = (K1_g*abs(error)+N1_g)*70.0+10.0
        elif -C1_g <= error < C1_g:    gamma_g = (K0_g*abs(error))*70.0+10.0
        elif C1_g <= error < C2_g:     gamma_g = (K1_g*error+N1_g)*70.0+10.0
        else:                           gamma_g = B2_g*70.0+10.0
        gamma_g_1 = gamma_g*np.pi/180.0
        ld_e = max(ld_min, min(abs(error)/np.cos(gamma_g_1), ld_max)); ld = ld_e
        B0,B1,B2 = 0.0,ld*np.sin(gamma_g_1),1.0
        C0,C1,C2 = 0.0,ld*np.cos(gamma_g_1),1.0
        m0,m1,m2 = a,1.0-a,a; eps = 1e-8
        K0 = 0.5*((B1-B0*m0)/(C1*m0-C0+abs(error)*(-m0+1.0)+eps)+(B0-B1*m1)/(C0*m1-C1+abs(error)*(-m1+1.0)+eps))
        K1 = 0.5*((B2-B1*m1)/(C2*m1-C1+abs(error)*(-m1+1.0)+eps)+(B1-B2*m2)/(C1*m2-C2+abs(error)*(-m2+1.0)+eps))
        N0 = 0.5*((B1*C0-B0*C1*m0)/(-C1*m0+C0+abs(error)*(m0-1.0)+eps)+(B0*C1-B1*C0*m1)/(-C0*m1+C1+abs(error)*(m1-1.0)+eps))
        N1 = 0.5*((B2*C1-B1*C2*m1)/(-C2*m1+C1+abs(error)*(m1-1.0)+eps)+(B1*C2-B2*C1*m2)/(-C1*m2+C2+abs(error)*(m2-1.0)+eps))
        if error <= -C2:        phi = (-B2*MAX_RAD)+(error+C2)*(K1*MAX_RAD)
        elif -C2 < error < -C1: phi = (K1*error-N1)*MAX_RAD
        elif -C1 <= error < C1: phi = (K0*error)*MAX_RAD
        elif C1 <= error < C2:  phi = (K1*error+N1)*MAX_RAD
        else:                   phi = (B2*MAX_RAD)+(error-C2)*(K1*MAX_RAD)
        return phi * self.fuzzy_params["k_out"]

    def reset(self):
        self.command_yaw.reset(); self.command_pitch.reset(); self.command_velocity.reset()
        self.velocity_d = 0.0; self.yaw_d = 0.0
        print(">>> LQR controller reset")

    def balance(self, data):
        a = self.get_command_pitch(data)
        c = self.get_command_yaw(data)
        left_torque  = -0.5*a - c
        right_torque =  0.5*a - c
        data.ctrl[self.actuator_ids["left_wheel"]]  = left_torque
        data.ctrl[self.actuator_ids["right_wheel"]] = right_torque


# --- VMC Controller ---
class VMC:
    def __init__(self, model):
        self.model = model
        self.act_ids = {
            "L1": model.actuator("L1_joint_ctrl").id,
            "L2": model.actuator("L2_joint_ctrl").id,
            "R1": model.actuator("R1_joint_ctrl").id,
            "R2": model.actuator("R2_joint_ctrl").id,
        }
        self.joint_handles = {}
        for name in ["L1", "L2", "R1", "R2"]:
            j_id = model.joint(f"{name}_joint").id
            self.joint_handles[name] = {
                "id": j_id, "qpos": model.jnt_qposadr[j_id], "dof": model.jnt_dofadr[j_id]
            }
        self.roll, self.roll_dot = 0.0, 0.0
        self.q = {"L1": 0.0, "L2": 0.0, "R1": 0.0, "R2": 0.0}
        self.v = {"L1": 0.0, "L2": 0.0, "R1": 0.0, "R2": 0.0}

    def update_states(self, data):
        quat = data.qpos[3:7]
        rpy  = euler.quat2euler(quat, axes='rzyx')
        self.roll = rpy[2]
        wx = data.qvel[3]; wy = data.qvel[4]; wz = data.qvel[5]
        cos_r = math.cos(self.roll); sin_r = math.sin(self.roll)
        cos_p = math.cos(rpy[1]); tan_p = math.tan(rpy[1])
        self.roll_dot = (wx + sin_r*tan_p*wy + cos_r*tan_p*wz) if abs(cos_p) > 1e-4 else wx
        for name, handle in self.joint_handles.items():
            self.q[name] = data.qpos[handle["qpos"]]
            self.v[name] = data.qvel[handle["dof"]]

    def vmc(self, data):
        k = 800; d = 2*k**0.5
        theta_10, theta_20 = 70*math.pi/180, 95*math.pi/180
        l1, l2 = 0.15, 0.25; x_d, z_d = 0.0353, 0.25
        z_d_comp = 1000*self.roll + 100*self.roll_dot
        q_l1,q_l2 = self.q["L1"],self.q["L2"]; v_l1,v_l2 = self.v["L1"],self.v["L2"]
        x_l = l1*math.sin(theta_10-q_l1)-l2*math.sin((theta_20+q_l2)-(theta_10-q_l1))
        z_l = l1*math.cos(theta_10-q_l1)+l2*math.cos((theta_20+q_l2)-(theta_10-q_l1))
        J_l = np.array([
            [-l1*math.cos(theta_10-q_l1)-l2*math.cos((theta_20+q_l2)-(theta_10-q_l1)), -l2*math.cos((theta_20+q_l2)-(theta_10-q_l1))],
            [ l1*math.sin(theta_10-q_l1)-l2*math.sin((theta_20+q_l2)-(theta_10-q_l1)), -l2*math.sin((theta_20+q_l2)-(theta_10-q_l1))]])
        p_dot_l = J_l @ np.array([v_l1,v_l2])
        F_l = np.array([k*(x_d-x_l)-d*p_dot_l[0], k*(z_d-z_l)-d*p_dot_l[1]-z_d_comp])
        tau_l = J_l.T @ F_l
        q_r1,q_r2 = self.q["R1"],self.q["R2"]; v_r1,v_r2 = self.v["R1"],self.v["R2"]
        x_r = l1*math.sin(theta_10+q_r1)-l2*math.sin((theta_20-q_r2)-(theta_10+q_r1))
        z_r = l1*math.cos(theta_10+q_r1)+l2*math.cos((theta_20-q_r2)-(theta_10+q_r1))
        J_r = np.array([
            [ l1*math.cos(theta_10+q_r1)+l2*math.cos((theta_20-q_r2)-(theta_10+q_r1)),  l2*math.cos((theta_20-q_r2)-(theta_10+q_r1))],
            [-l1*math.sin(theta_10+q_r1)+l2*math.sin((theta_20-q_r2)-(theta_10+q_r1)),  l2*math.sin((theta_20-q_r2)-(theta_10+q_r1))]])
        p_dot_r = J_r @ np.array([v_r1,v_r2])
        F_r = np.array([k*(x_d-x_r)-d*p_dot_r[0], k*(z_d-z_r)-d*p_dot_r[1]+z_d_comp])
        tau_r = J_r.T @ F_r
        data.ctrl[self.act_ids["L1"]] = tau_l[0]; data.ctrl[self.act_ids["L2"]] = tau_l[1]
        data.ctrl[self.act_ids["R1"]] = tau_r[0]; data.ctrl[self.act_ids["R2"]] = tau_r[1]


# ============================================================
# Data Logger
# ============================================================
class DataLogger:
    """
    Records data every LOG_EVERY steps during simulation.
    Calls plot() after GUI closes to draw four subplots.
    Recording rate = control rate / LOG_EVERY (default 500Hz / 10 = 50Hz).
    """
    def __init__(self, dt=0.002, log_every=10):
        self.dt        = dt
        self.log_every = log_every
        self._step     = 0          # internal counter for time axis

        # Data buffers (appended as list, converted to numpy at plot time)
        self.t            = []
        self.velocity     = []    # actual velocity (m/s)
        self.velocity_cmd = []    # velocity command (m/s)
        self.pitch_com    = []    # CoM lean angle (rad)
        self.pitch        = []    # actual pitch (rad)
        self.theta_pitch  = []    # inner-loop error (rad)
        self.roll         = []    # roll angle (rad)

    def record(self, lqr: 'LQR_Controller'):
        self._step += 1
        self.t.append(self._step * self.dt * self.log_every)
        self.velocity.append(lqr.robot_x_velocity)
        self.velocity_cmd.append(lqr.velocity_d)
        self.pitch_com.append(lqr.pitch_com)
        self.pitch.append(lqr.pitch)
        self.theta_pitch.append(lqr.theta_pitch)
        self.roll.append(lqr.roll)

    def plot(self):
        if len(self.t) == 0:
            print("[DataLogger] No data recorded, skip plotting.")
            return

        t            = np.array(self.t)
        velocity     = np.array(self.velocity)
        velocity_cmd = np.array(self.velocity_cmd)
        pitch_com    = np.degrees(np.array(self.pitch_com))
        pitch        = np.degrees(np.array(self.pitch))
        theta_pitch  = np.degrees(np.array(self.theta_pitch))
        roll         = np.degrees(np.array(self.roll))

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
        fig.suptitle('Robot Control Monitor', fontsize=13, y=0.99)

        # -- Plot 1: Velocity Tracking --
        ax = axes[0]
        ax.plot(t, velocity,     color='royalblue', linewidth=1.2, label='velocity (m/s)')
        ax.plot(t, velocity_cmd, color='tomato',    linewidth=1.5, linestyle='--', label='velocity_cmd (m/s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Tracking')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

        # -- Plot 2: pitch_com vs pitch --
        ax = axes[1]
        ax.plot(t, pitch_com, color='darkorange', linewidth=1.2, label='pitch_com (deg)  [CoM lean angle]')
        ax.plot(t, pitch,     color='steelblue',  linewidth=1.2, label='pitch (deg)      [actual pitch]')
        ax.set_ylabel('Angle (deg)')
        ax.set_title('pitch_com vs pitch')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

        # ── 图3：theta_pitch 内环误差 ──────────────────────────
        ax = axes[2]
        ax.plot(t, theta_pitch, color='mediumseagreen', linewidth=1.2, label='theta_pitch (deg)  [inner-loop error]')
        ax.set_ylabel('Angle (deg)')
        ax.set_title('theta_pitch  (inner-loop tracking error)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

        # ── 图4：roll ──────────────────────────────────────────
        ax = axes[3]
        ax.plot(t, roll, color='mediumpurple', linewidth=1.2, label='roll (deg)  [roll angle]')
        ax.set_ylabel('Angle (deg)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Roll')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

        plt.tight_layout()
        print(f"[DataLogger] {len(t)} data points, duration {t[-1]:.1f}s — showing plots...")
        plt.show()


# ============================================================
# Main
# ============================================================
def main():
    from mj_keyboard import KeyboardCommander

    model = mujoco.MjModel.from_xml_path(
        "/home/huang/wheel_leg/wheel_legged_genesis_new/assets/description/scence_Sim.xml")
    data  = mujoco.MjData(model)
    model.opt.timestep = 0.002

    vmc_ctrl = VMC(model)
    lqr_ctrl = LQR_Controller(model)
    cmd      = KeyboardCommander(v_step=0.1, yaw_step_deg=10.0)

    # Data logger: record every 10 physics steps (20ms) = 50Hz
    LOG_EVERY = 10
    logger    = DataLogger(dt=model.opt.timestep, log_every=LOG_EVERY)

    step = 0

    def controller(m, d):
        nonlocal step
        step += 1

        target_v, target_yaw = cmd.get_command()
        lqr_ctrl.velocity_d  = target_v
        lqr_ctrl.yaw_d       = target_yaw

        lqr_ctrl.update_imu_data(d)
        lqr_ctrl.update_joint_states(d)
        vmc_ctrl.update_states(d)

        vmc_ctrl.vmc(d)
        lqr_ctrl.balance(d)

        # Record every LOG_EVERY steps
        if step % LOG_EVERY == 0:
            logger.record(lqr_ctrl)

        # Terminal print (~once per second)
        if step % 500 == 0:
            print(f"pitch: {lqr_ctrl.pitch:.3f}  yaw: {lqr_ctrl.yaw:.3f}  roll: {lqr_ctrl.roll:.3f}")
            print(f"velocity: {lqr_ctrl.robot_x_velocity:.3f}  cmd: {lqr_ctrl.velocity_d:.3f}")
            print(f"pitch_com: {lqr_ctrl.pitch_com:.4f}  theta_pitch: {lqr_ctrl.theta_pitch:.4f}")
            print("-" * 50)

    mujoco.set_mjcb_control(controller)

    print("Simulation started, native GUI launched")
    print("Close the GUI window to auto-generate plots")

    # launch blocks until the GUI window is closed
    mujoco.viewer.launch(model, data)

    # After GUI closes
    mujoco.set_mjcb_control(None)
    print(f"\nGUI closed, generating plots...")
    logger.plot()


if __name__ == "__main__":
    main()