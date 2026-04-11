# 环境搭建指南

## 🎯 安装流程概览

```
1. 安装Ubuntu系统 → 2. 安装NVIDIA驱动 → 3. 安装Miniconda → 4. 创建虚拟环境 → 5. 安装项目依赖
```

---

## 1. 系统准备

### **1.1 安装Ubuntu 22.04**
- **视频教程**：
  - [一看就会！8分钟真机安装【Ubuntu/Windows】双系统](https://www.bilibili.com/video/BV1hL411r7p2)
  - [Windows11 安装 Ubuntu 避坑指南](https://www.bilibili.com/video/BV1Cc41127B9)
- **文字教程**：
  - [CSDN教程1](https://blog.csdn.net/m0_74115845/article/details/140880326)
  - [CSDN教程2](https://blog.csdn.net/2401_84064328/article/details/137232169)

**建议**：搭配AI工具或询问有经验的人协助安装。

### **1.2 安装NVIDIA显卡驱动**
- **教程**：
  - [CSDN教程](https://blog.csdn.net/a772304419/article/details/146601092)
  - [B站视频](https://b23.tv/4hEZrYz)

**关键步骤**：
```bash
# 查看显卡信息
lspci | grep -i nvidia

# 推荐使用ubuntu-drivers自动安装
sudo ubuntu-drivers autoinstall

# 重启后验证
nvidia-smi
```

### **1.3 安装Miniconda**
```bash
# 下载安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 运行安装
bash Miniconda3-latest-Linux-x86_64.sh

# 按照提示完成安装，重启终端
```

---

## 2. 项目环境搭建

### **2.1 创建虚拟环境**
```bash
# 创建Python 3.10环境（必须3.10）
conda create -n wl_new python=3.10 -y

# 激活环境
conda activate wl_new
```

### **2.2 安装核心依赖**

#### **必须安装的包**
```bash
# Genesis仿真器（必须0.2.1版本）
pip install genesis-world==0.2.1

# PyTorch（根据CUDA版本选择）
# 查看CUDA版本：nvidia-smi
# CUDA 12.6示例：
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 其他必要依赖
pip install transforms3d tensorboard pygame opencv-python
```

#### **项目内部依赖**
```bash
# 进入项目目录
cd wheel_legged_genesis_new

# 安装rsl-rl依赖（项目已包含）
cd rsl_rl && pip install -e . && cd ..

# 安装文件索引系统
python -c "import file_index; file_index.install_file_index()"
```

#### **MuJoCo安装（必要）**
```bash
# 建议查看官方文档或询问AI工具安装MuJoCo
# 通常需要：
# 1. 下载MuJoCo库文件
# 2. 设置环境变量
# 3. 安装mujoco-py
```

### **2.3 验证安装**
```bash
# 检查关键依赖
python -c "
import torch
import genesis
print('✅ PyTorch版本:', torch.__version__)
print('✅ CUDA可用:', torch.cuda.is_available())
print('✅ Genesis版本: 0.2.1')
"

# 运行简单测试
python locomotion/trainers/FZ_PID_train.py \
  --num_envs 256 \
  --max_iterations 5 \
  --exp_name env_test
```

---

## 3. 常见问题解决

### **3.1 版本兼容性问题**
```bash
# 如果Genesis版本错误
pip uninstall genesis-world -y
pip install genesis-world==0.2.1

# 如果出现导入错误
python -c "import file_index; file_index.install_file_index()"
```

### **3.2 CUDA/PyTorch版本不匹配**
```bash
# 查看CUDA版本
nvidia-smi
nvcc --version

# 根据CUDA版本重新安装PyTorch
# CUDA 12.x: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU版本: pip3 install torch torchvision torchaudio
```

### **3.3 依赖缺失**
```bash
# 使用pip检查并安装缺失的包
# 通常错误信息会提示缺少哪个包
pip install [缺失的包名]
```

---

## 4. AI辅助安装指南

### **遇到问题时的最佳实践**
1. **复制完整的错误信息**
2. **描述你的环境**：
   - 操作系统版本
   - CUDA版本（`nvidia-smi`输出）
   - Python版本（`python --version`）
   - 已执行的命令
3. **请求AI生成具体命令**

### **示例AI提示**
```
我在Ubuntu 22.04上安装轮腿机器人项目环境时遇到错误：

[粘贴完整的错误信息]

我的环境：
- 操作系统：Ubuntu 22.04
- CUDA版本：12.1（nvidia-smi显示）
- Python版本：3.10.12
- 已执行：conda create -n wl_new python=3.10 -y

请提供修复这个错误的完整命令。
```

---

## 5. 环境管理

### **5.1 常用命令**
```bash
# 环境管理
conda activate wl_new          # 激活环境
conda deactivate               # 退出环境
conda env list                 # 列出所有环境

# 依赖管理
pip freeze > requirements.txt  # 导出依赖
pip install -r requirements.txt # 安装依赖
```

### **5.2 重置环境**
```bash
# 如果环境损坏，重新创建
conda deactivate
conda env remove -n wl_new
conda create -n wl_new python=3.10 -y
conda activate wl_new

# 重新安装依赖
pip install genesis-world==0.2.1 torch transforms3d tensorboard
```

---

## 6. 安装完成验证

### **成功标志**
- ✅ `nvidia-smi`显示GPU信息
- ✅ `python -c "import torch; print(torch.cuda.is_available())"`返回True
- ✅ `python -c "import genesis"`不报错
- ✅ 训练脚本能正常启动

### **下一步**
环境搭建完成后，参考：
- [README.md](README.md) - 项目快速开始
- [GUIDE.md](GUIDE.md) - 核心使用指南

---

## 📝 注意事项

1. **版本严格性**：Python必须3.10，Genesis必须0.2.1
2. **环境隔离**：使用conda环境避免版本冲突
3. **AI工具**：复杂环境问题建议使用AI辅助解决
4. **逐步验证**：每步安装后验证是否成功

---

**提示**：环境安装是项目中最容易出问题的环节。如果遇到困难，不要犹豫，使用AI工具或寻求帮助。