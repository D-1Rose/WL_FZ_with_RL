import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def compute_flc(error_deg, domain_deg, gamma_deg, a_val, eth_val, k_out_val, ld_min_val, ld_max_val, fix_bug=True):
    """
    完全复刻 ST_SIT2_FLC_FM 的底层数学逻辑，并暴露所有核心变量
    """
    if fix_bug:
        error = error_deg / domain_deg 
        gamma_d_g = gamma_deg * np.pi / 180.0
        # 动态安全保护：确保 eth 永远小于 cos(gamma)
        eth = np.minimum(eth_val, np.cos(gamma_d_g) - 0.001)
        ld_min = ld_min_val
        ld_max = ld_max_val
    else:
        # 旧版包含BUG的代码：完全硬编码
        domain_deg = 15.0
        error = error_deg / domain_deg 
        gamma_d_g = 75.0 * np.pi / 180.0
        eth = 0.06
        ld_min = 0.3
        ld_max = 0.9


    # ---------- ld_g计算 ----------
    ld_g = (np.cos(gamma_d_g) - eth) / np.cos(gamma_d_g)

    B0_g, B2_g = 0.0, 1.0
    B1_g = ld_g * np.sin(gamma_d_g)
    C0_g, C2_g = 0.0, 1.0
    C1_g = ld_g * np.cos(gamma_d_g)

    K0_g = B1_g / C1_g if C1_g != 0 else 0
    K1_g = (B2_g - B1_g) / (C2_g - C1_g)
    N0_g = 0.0
    N1_g = (B1_g * C2_g - B2_g * C1_g) / (C2_g - C1_g)

    # ---------- gamma_g 分段计算 ----------
    cond1 = (error <= -C2_g)
    cond2 = (error > -C2_g) & (error < -C1_g)
    cond3 = (error >= -C1_g) & (error < C1_g)
    cond4 = (error >= C1_g) & (error < C2_g)
    cond5 = (error >= C2_g)

    gamma_g = np.zeros_like(error)
    gamma_g = np.where(cond1, B2_g * 70 + 10, gamma_g)
    gamma_g = np.where(cond2, (K1_g * np.abs(error) + N1_g) * 70 + 10, gamma_g)
    gamma_g = np.where(cond3, (K0_g * np.abs(error)) * 70 + 10, gamma_g)
    gamma_g = np.where(cond4, (K1_g * error + N1_g) * 70 + 10, gamma_g)
    gamma_g = np.where(cond5, B2_g * 70 + 10, gamma_g)

    # ---------- ld_e 与隶属度参数 ----------
    gamma_g_1 = gamma_g * np.pi / 180.0
    ld_e = np.abs(error) / np.cos(gamma_g_1)
    
    # 核心：ld_min 和 ld_max 发生作用的地方
    ld_e = np.clip(ld_e, ld_min, ld_max)
    ld = ld_e

    B0 = np.zeros_like(ld)
    B1 = ld * np.sin(gamma_g_1)
    B2 = np.ones_like(ld)
    C0 = np.zeros_like(ld)
    C1 = ld * np.cos(gamma_g_1)
    C2 = np.ones_like(ld)

    m0 = m2 = a_val
    m1 = 1 - a_val

    abs_err = np.abs(error)
    
    # 避免除以0的极小偏置
    eps = 1e-8
    
    K0 = 0.5 * ((B1 - B0 * m0) / (C1 * m0 - C0 + abs_err * (-m0 + 1) + eps) +
                (B0 - B1 * m1) / (C0 * m1 - C1 + abs_err * (-m1 + 1) + eps))
    K1 = 0.5 * ((B2 - B1 * m1) / (C2 * m1 - C1 + abs_err * (-m1 + 1) + eps) +
                (B1 - B2 * m2) / (C1 * m2 - C2 + abs_err * (-m2 + 1) + eps))
    N0 = 0.5 * ((B1 * C0 - B0 * C1 * m0) / (-C1 * m0 + C0 + abs_err * (m0 - 1) + eps) +
                (B0 * C1 - B1 * C0 * m1) / (-C0 * m1 + C1 + abs_err * (m1 - 1) + eps))
    N1 = 0.5 * ((B2 * C1 - B1 * C2 * m1) / (-C2 * m1 + C1 + abs_err * (m1 - 1) + eps) +
                (B1 * C2 - B2 * C1 * m2) / (-C1 * m2 + C2 + abs_err * (m2 - 1) + eps))

    phi = np.zeros_like(error)
    
    if fix_bug:
        # 新版修复后：严谨的量纲对齐与外推
        MAX_RAD = domain_deg / 180.0 * np.pi
        phi = np.where(cond1, (-B2 * MAX_RAD) + (error + C2) * (K1 * MAX_RAD), phi)
        phi = np.where(cond2, (K1 * error - N1) * MAX_RAD, phi)
        phi = np.where(cond3, (K0 * error) * MAX_RAD, phi)
        phi = np.where(cond4, (K1 * error + N1) * MAX_RAD, phi)
        phi = np.where(cond5, (B2 * MAX_RAD) + (error - C2) * (K1 * MAX_RAD), phi)
        return phi * k_out_val * 180.0 / np.pi 
    else:
        # 旧版代码逻辑
        phi = np.where(cond1, -B2*15/180*np.pi + (error + C2), phi)
        phi = np.where(cond2, (K1 * error - N1)*15/180*np.pi, phi)
        phi = np.where(cond3, (K0 * error)*15/180*np.pi, phi)
        phi = np.where(cond4, (K1 * error + N1)*15/180*np.pi, phi)
        phi = np.where(cond5, B2*15/180*np.pi + (error - C2), phi)
        return phi * 180.0 / np.pi 


# ================== UI 布局与初始化 ==================
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.5) # 给底部留出超大空间放7个滑块
error_x = np.linspace(-40, 40, 600) 

# 初始绘制
line_old, = ax.plot(error_x, compute_flc(error_x, 15.0, 75.0, 0.5, 0.06, 1.0, 0.3, 0.9, False), 
                    'r--', lw=2, alpha=0.6, label='Old Buggy FLC (Domain=15)')
line_new, = ax.plot(error_x, compute_flc(error_x, 30.0, 45.0, 0.5, 0.06, 1.0, 0.3, 0.9, True), 
                    'b-', lw=3, label='New Fixed FLC (Adjustable)')

ax.set_title('Ultimate Type-2 Fuzzy Logic Surface Explorer', fontsize=16, fontweight='bold')
ax.set_xlabel('Physical Pitch Error (Degrees)', fontsize=12)
ax.set_ylabel('Output Compensation Angle (Degrees)', fontsize=12)
ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=1)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend()
ax.set_ylim(-60, 60)

# ================== 定义 7 个 Slider 区域 ==================
slider_color = 'lightgoldenrodyellow'
ax_domain = plt.axes([0.15, 0.40, 0.75, 0.02], facecolor=slider_color)
ax_gamma  = plt.axes([0.15, 0.35, 0.75, 0.02], facecolor=slider_color)
ax_a      = plt.axes([0.15, 0.30, 0.75, 0.02], facecolor=slider_color)
ax_eth    = plt.axes([0.15, 0.25, 0.75, 0.02], facecolor=slider_color)
ax_ldmin  = plt.axes([0.15, 0.20, 0.75, 0.02], facecolor=slider_color)
ax_ldmax  = plt.axes([0.15, 0.15, 0.75, 0.02], facecolor=slider_color)
ax_kout   = plt.axes([0.15, 0.10, 0.75, 0.02], facecolor=slider_color)

sl_domain = Slider(ax_domain, 'Domain Deg', 10.0, 60.0, valinit=30.0)
sl_gamma  = Slider(ax_gamma,  'Gamma Deg',  30.0, 80.0, valinit=45.0)
sl_a      = Slider(ax_a,      'Param a',    0.01, 0.99, valinit=0.5)
sl_eth    = Slider(ax_eth,    'eth',        0.00, 0.50, valinit=0.06)
sl_ldmin  = Slider(ax_ldmin,  'ld_min',     0.00, 0.80, valinit=0.3)
sl_ldmax  = Slider(ax_ldmax,  'ld_max',     0.50, 2.00, valinit=0.9)
sl_kout   = Slider(ax_kout,   'K_out',      0.50, 4.00, valinit=1.0)

def update(val):
    d_val = sl_domain.val
    g_val = sl_gamma.val
    a_val = sl_a.val
    e_val = sl_eth.val
    lmin_val = sl_ldmin.val
    lmax_val = sl_ldmax.val
    k_val = sl_kout.val
    
    # 更新新版曲线
    new_y = compute_flc(error_x, d_val, g_val, a_val, e_val, k_val, lmin_val, lmax_val, True)
    line_new.set_ydata(new_y)
    fig.canvas.draw_idle()

sl_domain.on_changed(update)
sl_gamma.on_changed(update)
sl_a.on_changed(update)
sl_eth.on_changed(update)
sl_ldmin.on_changed(update)
sl_ldmax.on_changed(update)
sl_kout.on_changed(update)

plt.show()