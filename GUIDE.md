# 详细使用指南

## 文档概述

本指南提供轮腿机器人项目的完整使用说明，涵盖从基础操作到高级定制的所有内容。无论你是初次使用还是需要修改项目，都能在这里找到所需信息。

---

## 一、快速验证与基础操作

### 1.1 环境验证
在开始任何操作前，请确认环境已正确配置：

```bash
# 1. 激活conda环境
conda activate wl_new

# 2. 进入项目目录
cd ~/wheel_leg/wheel_legged_genesis_new

# 3. 验证核心依赖
python -c "
import torch
import genesis
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print('Genesis 0.2.1: 安装成功')
"
```

### 1.2 最小化测试运行
运行一个简短的训练以验证整个流程能正常工作：

```bash
# 使用最小配置进行测试（约1-2分钟）
python locomotion/trainers/FZ_PID_train.py \
  --num_envs 512 \
  --max_iterations 10 \
  --exp_name verification_test
```

**预期输出**：
- 训练正常启动，无导入错误
- 每步迭代显示奖励信息
- 在`locomotion/logs/verification_test/`目录生成日志文件

### 1.3 监控训练进度
```bash
# 启动TensorBoard监控
tensorboard --logdir locomotion/logs/verification_test 

# 浏览器访问 http://localhost:6006 查看：
# - reward/total_reward: 总奖励曲线
# - losses/value_loss: 值函数损失
# - actions/action_mean: 动作均值分布
```

---

## 二、标准工作流程详解

### 2.1 完整训练流程

#### 步骤1：启动标准训练
```bash
# 标准配置：4096个并行环境，300次迭代
python locomotion/trainers/FZ_PID_train.py \
  --num_envs 4096 \
  --max_iterations 300 \
  --exp_name standard_experiment
```

**参数说明**：
- `--num_envs`: 并行环境数量，影响训练速度和GPU内存使用
- `--max_iterations`: 训练迭代次数，每次迭代包含多个环境步
- `--exp_name`: 实验名称，用于日志目录命名

#### 步骤2：训练过程说明
训练过程中：
1. **自动保存**：每25次迭代保存一次模型检查点（`model_{iter}.pt`）
2. **配置保存**：训练配置保存为`cfgs.pkl`
3. **日志记录**：TensorBoard日志实时更新

#### 步骤3：训练输出文件
```
locomotion/logs/standard_experiment/
├── model_25.pt          # 第25次迭代的检查点
├── model_50.pt          # 第50次迭代的检查点
├── ...                 # 每25步保存一次
├── model_300.pt        # 最终模型
├── cfgs.pkl            # 训练配置（pickle格式）
└── events.out.tfevents.* # TensorBoard日志文件
```

### 2.2 模型评估与导出

#### 步骤1：选择检查点进行评估
```bash
# 查看可用的检查点
ls locomotion/logs/standard_experiment/model_*.pt

# 通常选择最终检查点（300次迭代）
python locomotion/evaluators/FZ_PID_eval.py \
  --exp_name standard_experiment \
  --ckpt 300
```

#### 步骤2：评估过程说明
评估脚本执行以下操作：
1. 加载指定检查点的模型
2. 在Genesis仿真环境中运行评估
3. 导出融合模型（包含归一化层）

#### 步骤3：评估输出文件
```
locomotion/logs/standard_experiment_eval/
└── policy_fused.pt    # 融合模型，用于MuJoCo验证
```

### 2.3 MuJoCo验证测试

#### 步骤1：使用RL策略验证
```bash
# 使用导出的融合模型进行MuJoCo验证
python locomotion/evaluators/mj_eval.py --RL \
  --model locomotion/logs/standard_experiment_eval/policy_fused.pt \
  --log_name mujoco_validation
```

#### 步骤2：纯控制器测试（对比基准）
```bash
# 不使用RL策略，仅测试基础控制器
python locomotion/evaluators/mj_eval.py --log_name baseline_pid
```

#### 步骤3：验证参数说明
- `--RL`: 启用RL参数自适应模式
- `--model`: 指定RL模型文件路径（默认：`policy_fused.pt`）
- `--log_name`: 日志名称前缀，用于结果目录命名

#### 步骤4：验证输出文件
```
logs/mujoco_comparison/mujoco_validation_[timestamp]/
├── metrics.csv          # 性能指标CSV文件
├── Analysis_Plot_1.png  
├── Analysis_Plot_2.png  
├── Analysis_Plot_3.png  
├── Analysis_Plot_4.png  
└── events.out.tfevents.* # TensorBoard日志
```

---

## 三、项目核心文件详解

### 3.1 训练系统文件

#### `locomotion/trainers/FZ_PID_train.py`
**功能**：主训练脚本，实现PPO算法训练流程
**关键部分**：
- `get_cfgs()`: 返回所有训练配置（环境、奖励、地形等）
- 训练循环：管理迭代、数据收集、模型更新
- 检查点保存：每25步自动保存模型

**使用场景**：
- 修改训练参数（学习率、批量大小等）
- 调整奖励函数权重
- 更改地形配置

#### 配置结构示例：
```python
def get_cfgs():
    return {
        "train_cfg": { ... },      # 训练算法配置
        "env_cfg": { ... },        # 环境参数配置
        "reward_cfg": { ... },     # 奖励函数配置
        "terrain_cfg": { ... },    # 地形生成配置
        # ... 其他配置
    }
```

### 3.2 评估系统文件

#### `locomotion/evaluators/FZ_PID_eval.py`
**功能**：Genesis环境模型评估与融合模型导出
**关键流程**：
1. 加载训练好的模型检查点
2. 在Genesis环境中运行评估episode
3. 导出包含归一化层的融合模型

#### `locomotion/evaluators/mj_eval.py`
**功能**：MuJoCo环境验证与性能分析
**关键特性**：
- 支持RL策略和纯控制器两种模式
- 自动数据记录与图表生成
- 实时性能监控

### 3.3 环境定义文件

#### `locomotion/environments/FZ_PID_env_c.py`
**功能**：轮腿机器人环境实现
**核心组件**：
- `WheelLeggedEnv`类：环境主类
- `step()`方法：环境步进，包含RL动作到控制器参数的映射
- 观测空间：机器人状态、传感器数据
- 奖励计算：根据配置计算每一步的奖励

**关键函数**：
```python
def step(self, actions):
    # RL动作映射到控制器参数
    self.lqr_controller.params['gamma_d_g'] = self.scale_param(
        actions[:, 0], 75.0*math.pi/180.0, 45.0*math.pi/180.0, 89.0*math.pi/180.0, factor=0.3
    )
    # ... 其他参数映射
```

### 3.4 控制器实现文件

#### `locomotion/controllers/FZ_PID_c.py`
**功能**：模糊PID控制器实现（项目中称为LQR_Controller）
**核心类**：
- `BatchedLQRController`: 批量处理的模糊PID控制器（轮子）
- `BatchedVMC`: vmc模型控制器（腿）

**控制器参数**：
```python
# 默认参数值
self.params['gamma_d_g'] = 75.0 * math.pi / 180.0  # γ角度，默认75°
self.params['eth'] = 0.05                          # eth参数，默认0.05
self.params['a'] = 0.05                            # a参数，默认0.05
```

### 3.5 机器人模型文件

#### `assets/description/urdf/scence.xml`
**功能**：MuJoCo机器人模型定义
**文件结构**：
- `<mujoco>`: 根元素，定义模型属性
- `<worldbody>`: 世界和机器人几何体定义
- `<actuator>`: 执行器（电机）定义
- `<sensor>`: 传感器定义（可选）

**关键元素**：
```xml
<!-- 机器人基座 -->
<body name="base_link" pos="0 0 0.3">
  <joint name="free_joint" type="free"/>
  <geom name="base" type="box" size="0.15 0.1 0.05"/>
  <!-- 腿部定义 -->
</body>

<!-- 执行器定义 -->
<actuator>
  <motor name="L1_joint" joint="L1_joint" ctrlrange="-1 1"/>
  <!-- 其他执行器 -->
</actuator>
```

---

## 四、配置修改指南

### 4.1 训练参数修改

#### 修改位置：`FZ_PID_train.py` → `get_cfgs()`函数

#### 4.1.1 算法参数(一般不用改)
```python
train_cfg = {
    "algorithm": {
        "learning_rate": 1e-4,           # 学习率，调整训练速度
        "num_learning_epochs": 3,        # 每次迭代的学习周期数
        "num_mini_batches": 5,           # 小批量数量
        "clip_range": 0.2,               # PPO裁剪范围
        "entropy_coef": 0.01,            # 熵系数，鼓励探索
        "value_loss_coef": 0.5,          # 值函数损失权重
    }
}
```

**调整建议**：
- 训练不稳定时：降低`learning_rate`（如5e-5）
- 需要更多探索：增加`entropy_coef`（如0.02）
- 值函数拟合差：增加`value_loss_coef`（如1.0）

#### 4.1.2 运行参数
```python
train_cfg = {
    "runner": {
        "algorithm_class_name": "PPO",
        "max_iterations": 300,           # 最大迭代次数
        "save_interval": 25,             # 检查点保存间隔
        "experiment_name": args.exp_name,
    }
}
```

### 4.2 奖励函数修改

#### 奖励权重修改位置：`FZ_PID_train.py` → `reward_cfg`
```python
reward_cfg = {
    "tracking_lin_sigma": 0.10,         # 线速度跟踪容差
    "inner_pitch_sigma": 0.08,          # 内部俯仰角容差
    "reward_scales": {
        # 正向奖励项
        "tracking_lin_vel": 15.0,       # 线速度跟踪奖励
        "inner_pitch_error": 2.5,       # 内部俯仰角误差奖励
        "pitch": 5,                     # 俯仰角奖励
        
        # 负向惩罚项
        "action_norm": -1,              # 动作范数惩罚
        "action_rate": -4.5,            # 动作变化率惩罚
        "roll_vel": -0.5,               # 滚转角速度惩罚
    }
}
```
#### 奖励函数修改位置：`FZ_PID_env.py` → `最底下的奖励函数设计`


**调整策略**：
1. **强调速度跟踪**：增加`tracking_lin_vel`权重
2. **强调稳定性**：增加`pitch`和`inner_pitch_error`权重
3. **减少能量消耗**：增加`action_norm`惩罚
4. **平滑控制**：增加`action_rate`惩罚
#### 可以查阅一下资料或ai，如果能用别人设计好的效果不错的针对轮足机器人运动学动力学模型的多目标优化函数的话，将其作为奖励函数是很不错的

### 4.3 地形配置修改

#### 修改位置：`FZ_PID_train.py` → `terrain_cfg`（正常来说不用修改这里）
```python
terrain_cfg = {
    "terrain": True,                    # 是否启用地形
    "eval": "slope_10deg_fit",          # 评估地形类型
    "horizontal_scale": 0.05,           # 水平缩放系数
    "vertical_scale": 0.005,            # 垂直缩放系数
    "border_size": 20,                  # 地形边界大小
    "num_rows": 10,                     # 地形行数
    "num_cols": 20,                     # 地形列数
    "curriculum": True,                 # 是否启用课程学习
}
```

**可用地形类型**：
- `"flat"`: 平原地形
- `"slope_10deg_fit"`: 10度斜坡地形
- `"stairs"`: 楼梯地形
- `"rough"`: 粗糙地形
#### 还有许多地形在asset文件夹中，可以按需要查看也可以自己生成，注意要明确如何使用生成的png地形图片

**缩放参数说明**：
- `horizontal_scale`: 控制地形特征的宽度，值越小特征越精细
- `vertical_scale`: 控制地形特征的高度，值越大地形越陡峭

### 4.4 环境参数修改

#### 修改位置：`FZ_PID_train.py` → `env_cfg`

```python
env_cfg = {
    "num_envs": args.num_envs,          # 环境数量（从命令行参数获取）
    "env_spacing": 5.0,                 # 环境间距离
    "send_timeouts": True,              # 是否发送超时信号
    "episode_length_s": 20.0,           # 每个episode的长度（秒）
    "num_actions": 3,                   # 动作空间维度
    "dof_names": [                      # 关节名称列表
        "L1_joint", "L2_joint", "R1_joint",
        "R2_joint", "L3_joint", "R3_joint"
    ],
}
```

### 4.5 控制器参数范围修改

#### 修改位置：`FZ_PID_env_c.py` → `step()`函数（这个函数代表了每一次仿真步进做了什么，想要了解具体的仿真流程一定要熟悉里面的内容!）

```python
# RL动作到控制器参数的映射
self.lqr_controller.params['gamma_d_g'] = self.scale_param(
    real_actions[:, 0], 
    75.0 * math.pi / 180.0,  # 默认值：75°
    45.0 * math.pi / 180.0,  # 最小值：45°
    89.0 * math.pi / 180.0,  # 最大值：89°
    factor=0.3               # 敏感度因子
)

self.lqr_controller.params['eth'] = self.scale_param(
    real_actions[:, 1],
    0.05,     # 默认值
    0.01,     # 最小值
    0.1,      # 最大值
    factor=0.2
)

self.lqr_controller.params['a'] = self.scale_param(
    real_actions[:, 2],
    0.05,     # 默认值
    0.01,     # 最小值
    0.99,     # 最大值
    factor=0.2
)
```

**参数说明**：
- `factor`: 控制RL动作对参数的影响程度，值越大RL调整范围越大
- 修改范围后需要重新训练模型

### 4.6 机器人模型修改

#### 修改位置：`assets/description/urdf/scence.xml`(scence.xml为场景文件，文件中引用了地形的png图片路径，还有机器人的xml文件)



####  修改关节限制（按需求去修改）
```xml
<!-- 修改关节活动范围 -->
<joint name="L1_joint" type="hinge" axis="0 1 0" range="-0.6 0.6"/>
<!-- 原范围：range="-0.5 0.5" -->

<!-- 修改执行器力矩范围 -->
<actuator>
  <motor name="L1_joint" joint="L1_joint" ctrlrange="-1.2 1.2"/>
  <!-- 原范围：ctrlrange="-1 1" -->
</actuator>
```


---

## 五、关键注意事项

### 5.1 必须保持不变的参数

#### 仿真时间步长
```python
# 在 mj_eval.py 中定义
sim_dt = 0.002  # 500Hz控制频率
```

**重要性**：此参数与机器人硬件控制频率匹配，修改会导致控制不稳定。

#### 控制频率
```python
control_decimation = 10  # RL控制频率为50Hz（500Hz/10）
```

### 5.2 版本兼容性要求

#### 必须使用的版本
- **Python**: 3.10.x（必须，其他版本不兼容）
- **Genesis**: 0.2.1（必须，`pip install genesis-world==0.2.1`）
- **PyTorch**: 根据CUDA版本选择对应版本

#### 验证版本兼容性
```bash
# 检查关键依赖版本
python -c "
import sys
print(f'Python: {sys.version.split()[0]}')
import torch
print(f'PyTorch: {torch.__version__}')
import genesis
print('Genesis: 0.2.1 (确认安装成功)')
"
```

### 5.3 文件索引系统（这里可有可无，只要不影响主要脚本运行就没事）

项目使用`file_index.py`处理模块导入重定向，确保文件移动后原有导入仍然有效。

#### 常见操作
```bash
# 安装文件索引系统
python -c "import file_index; file_index.install_file_index()"

# 查看当前映射
python -c "import file_index; print('当前映射:', file_index.FILE_MAPPINGS)"
```

#### 添加新映射
```python
# 在 file_index.py 的 FILE_MAPPINGS 字典中添加
FILE_MAPPINGS = {
    # 现有映射...
    "old_module.path": "new.module.path",
    "custom_module": "locomotion.utils.custom_module",
}
```

---

## 六、高级操作与调试（可搭配ai进行尝试）

### 6.1 批量实验管理

#### 批量训练不同配置
```bash
# 使用脚本批量运行不同配置
for envs in 2048 4096 8192; do
    python locomotion/trainers/FZ_PID_train.py \
        --num_envs $envs \
        --max_iterations 300 \
        --exp_name "envs_${envs}_test"
done
```

#### 批量评估多个检查点
```bash
# 评估一个实验的所有重要检查点
for iter in 50 100 150 200 250 300; do
    python locomotion/evaluators/FZ_PID_eval.py \
        --exp_name my_experiment \
        --ckpt $iter
done
```

### 6.2 训练监控与调试

#### 实时监控命令
```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控系统资源
htop

# 监控训练日志
tail -f locomotion/logs/experiment_name/train.log
```

#### TensorBoard使用
```bash
# 同时监控多个实验
tensorboard --logdir locomotion/logs/ --port 6006

# 在TensorBoard中比较不同实验：
# 1. 打开Scalars标签页
# 2. 选择要比较的指标（如reward/total_reward）
# 3. 勾选不同实验的日志进行对比
```



### 6.3 性能优化建议

#### 训练速度优化
1. **增加并行环境数**：在GPU内存允许范围内最大化`--num_envs`
2. **调整批量大小**：根据硬件调整`num_mini_batches`
3. **使用混合精度训练**：在支持的情况下启用FP16训练
4. **优化数据加载**：确保数据预处理不成为瓶颈

#### 内存使用优化
```bash
# 监控GPU内存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 如果内存不足，减少环境数
python locomotion/trainers/FZ_PID_train.py --num_envs 2048 ...
```

### 6.4 实验记录与管理

#### 实验命名规范
```
# 建议的命名格式
exp_base_001              # 基础实验
exp_terrain_stairs_01     # 楼梯地形实验
exp_model_heavy_01        # 重型模型实验
exp_reward_speed_01       # 速度优化实验
exp_lr_1e5_01            # 学习率实验
```

#### 实验记录模板
```markdown
# 实验记录: exp_terrain_stairs_01

## 实验配置
- 日期: 2026-04-10
- 训练脚本: FZ_PID_train.py
- 参数: --num_envs 4096 --max_iterations 300
- 地形配置: stairs, horizontal_scale=0.025

## 修改内容
1. 地形类型改为stairs
2. horizontal_scale调整为0.025

## 结果
- 最终奖励: [填写数值]
- 训练时间: [填写时间]
- 关键观察: [填写观察结果]

## 下一步
- [ ] 尝试不同的horizontal_scale值
- [ ] 对比stairs和平地形的性能差异
```


---

## 七、扩展与定制开发

### 7.1 添加新的地形类型

#### 步骤1：准备地形高度图
1. 创建灰度PNG图像（建议512×512或1024×1024）
2. 白色（255）表示最高点，黑色（0）表示最低点
3. 保存到地形目录（如`terrain_heightmaps/custom_terrain.png`）

#### 步骤2：修改地形生成代码

在asset/terrain中有生成地形的脚本，可以按照需求自己修改

#### 步骤3：配置地形参数
```python
# 在训练配置中添加
terrain_cfg = {
    "terrain": True,
    "eval": "custom_terrain",  # 使用自定义地形
    "horizontal_scale": 0.03,
    "vertical_scale": 0.006,
    # ... 其他参数
}
```

### 7.2 添加新的奖励项

#### 步骤1：定义奖励计算函数
```python
# 在 FZ_PID_env_c.py 的 _compute_rewards() 函数中修改，这里是所有奖励函数的总计算方式
def _compute_rewards(self):
    # 现有奖励计算...
    
    # 添加新的奖励项
    energy_efficiency = self._compute_energy_efficiency()
    rewards += self.reward_cfg["reward_scales"].get("energy_efficiency", 0.0) * energy_efficiency
    
    return rewards

```

#### 步骤2：配置奖励权重
```python
reward_cfg = {
    "reward_scales": {
        # 现有奖励项...
        "energy_efficiency": 0.1,  # 新增的能量效率奖励
    }
}
```

### 7.3 修改网络结构

#### 步骤1：定位网络定义
```python
# 在训练脚本中查找策略网络和价值网络定义
# 通常位于模型初始化部分

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 现有网络结构
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
```

#### 步骤2：修改网络结构
```python
# 示例：增加网络层数和宽度
class EnhancedPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 更深的网络结构
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)
        
        # 添加Dropout防止过拟合
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return torch.tanh(self.fc5(x))
```

### 7.4 集成新的控制器算法

#### 步骤1：创建新控制器类
```python
# 在 controllers/ 目录下创建新文件，如 advanced_controller.py
import torch
import numpy as np

class AdvancedController:
    def __init__(self, num_envs, device=None):
        self.num_envs = num_envs
        self.device = device or torch.device('cpu')
        
        # 控制器参数
        self.params = {
            'param1': torch.full((num_envs,), 0.5, device=device),
            'param2': torch.full((num_envs,), 0.3, device=device),
        }
        
    def update(self, observations, commands):
        """根据观测和指令更新控制输出"""
        # 实现控制算法
        control_output = self._compute_control(observations, commands)
        return control_output
    
    def _compute_control(self, observations, commands):
        """具体的控制计算逻辑"""
        # 实现具体的控制算法
        pass
```

#### 步骤2：集成到环境中
```python
# 在 FZ_PID_env_c.py 中导入并使用新控制器
from locomotion.controllers.advanced_controller import AdvancedController

class WheelLeggedEnv:
    def __init__(self, ...):
        # 现有初始化代码...
        
        # 添加新控制器
        self.advanced_controller = AdvancedController(
            num_envs=self.num_envs,
            device=self.device
        )
    
    def step(self, actions):
        # 在适当的位置调用新控制器
        if self.use_advanced_controller:
            control_output = self.advanced_controller.update(
                observations=self._get_observations(),
                commands=self.commands
            )
            # 应用控制输出...
```

---
