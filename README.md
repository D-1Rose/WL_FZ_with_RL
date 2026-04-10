# 轮腿机器人强化学习平台

## 🎯 项目简介

通过强化学习优化模糊PID控制器的三个关键参数（γ, eth, a），实现轮腿机器人的稳定高速运动。

**核心流程**：Genesis训练 → 模型导出 → MuJoCo验证

---

## 📚 文档导航

1. **[SETUP.md](SETUP.md)** - 环境搭建指南（先看这个！）
2. **[GUIDE.md](GUIDE.md)** - 核心使用指南
3. **本文档** - 项目总览 + 快速开始

---

## 🚀 快速开始

### **前提条件**
- 已完成环境搭建（参考[SETUP.md](SETUP.md)）
- 已激活虚拟环境：`conda activate wl_new`

### **5分钟测试**
```bash
# 1. 运行简短训练
python locomotion/trainers/FZ_PID_train.py \
  --num_envs 1024 \
  --max_iterations 10 \
  --exp_name quick_test

# 2. 监控训练进度
tensorboard --logdir locomotion/logs/quick_test
# 访问 http://localhost:6006
```

### **完整工作流程**
```bash
# 1. 训练（Genesis）
python locomotion/trainers/FZ_PID_train.py \
  --num_envs 4096 \
  --max_iterations 300 \
  --exp_name my_exp

# 2. 评估导出
python locomotion/evaluators/FZ_PID_eval.py \
  --exp_name my_exp \
  --ckpt 300

这里可以使用已经训练好的模型进行直接测试：
python locomotion/evaluators/FZ_PID_eval.py \
  --exp_name my_exp \
  --ckpt 300


# 3. 验证（MuJoCo）
python locomotion/evaluators/mj_eval.py --RL \
  --model locomotion/logs/my_exp_eval/policy_fused.pt \
  --log_name validation


这里同样可以使用已经训练好的模型进行直接测试：
python locomotion/evaluators/mj_eval.py --RL \
  --model locomotion/logs/test_eval/2026-04-09_23-20-35_ckpt700/policy_fused.pt/policy_fused.pt \
  --log_name validation
```

---

## 📁 项目结构

```
wheel_legged_genesis_new/
├── locomotion/           # 核心代码
│   ├── trainers/        # 训练脚本
│   ├── evaluators/      # 评估脚本
│   ├── environments/    # 环境定义
│   ├── controllers/     # 控制器
│   └── logs/           # 训练日志
├── assets/              # 资源文件
├── file_index.py       # 文件索引系统
└── [文档]
```

**核心文件**：
- 训练：`locomotion/trainers/FZ_PID_train.py`
- 评估：`locomotion/evaluators/FZ_PID_eval.py`
- 验证：`locomotion/evaluators/mj_eval.py`
- 环境：`locomotion/environments/FZ_PID_env_c.py`
- 控制器：`locomotion/controllers/FZ_PID_c.py`

---

## ⚠️ 重要提示

### **必须遵守**
- **Python版本**：必须使用3.10
- **Genesis版本**：必须使用0.2.1（`pip install genesis-world==0.2.1`）
- **时间步长**：`sim_dt=0.002`（500Hz，**绝对不要修改**）

### **控制器参数范围**
```python
# RL动作到控制器参数的映射（硬编码）：
# action[0] → gamma_d_g: [45°, 89°]  (默认75°)
# action[1] → eth: [0.01, 0.1]       (默认0.05)
# action[2] → a: [0.01, 0.99]        (默认0.05)
```

### **文件索引系统**
如果出现导入错误：
```bash
python -c "import file_index; file_index.install_file_index()"
```

---

## 🔧 输出文件说明

### **训练输出**
```
locomotion/logs/{exp_name}/
├── model_*.pt          # 检查点（每25步保存）
├── cfgs.pkl           # 训练配置
└── events.out.tfevents.*  # TensorBoard日志
```

### **评估输出**
```
locomotion/logs/{exp_name}_eval/
└── policy_fused.pt    # 融合模型（用于MuJoCo）
```

### **验证输出**
```
logs/mujoco_comparison/{log_name}_*/
├── metrics.csv        # 性能指标
└── Analysis_Plot_*.png # 分析图表
```

---

## 🎯 标准配置

### **训练参数**
- 并行环境数：4096（推荐）
- 训练迭代数：300（推荐）
- 保存间隔：25步（自动保存检查点）

### **关键配置（硬编码）**
```python
# 训练配置
learning_rate = 1e-4
save_interval = 25

# 仿真参数
sim_dt = 0.002          # 500Hz控制频率
FINISH_LINE_X = 7.5     # MuJoCo验证终点线
```

---

## 📞 获取帮助

### **环境问题**
1. 参考 [SETUP.md](SETUP.md) 中的详细指南
2. 使用AI工具辅助解决依赖问题
3. 复制错误信息寻求帮助

### **使用问题**
1. 参考 [GUIDE.md](GUIDE.md) 中的完整指南
2. 检查命令格式和参数
3. 验证文件路径是否正确

### **代码问题**
1. 查看脚本注释和实现
2. 检查文件索引系统是否安装
3. 确保版本兼容性

---

## 📄 许可证

MIT License

---

**文档版本**：2026-04-10  
**项目版本**：FZ_PID核心版本  
**适用系统**：Ubuntu 22.04 + Python 3.10