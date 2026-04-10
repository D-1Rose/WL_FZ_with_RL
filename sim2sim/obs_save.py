import os
import torch
from typing import Dict, Union
import numpy as np

class TensorTypeSaver:
    def __init__(self, save_dir: str = "./data", device: str = "cpu"):
        """
        初始化数据保存器
        :param save_dir: 数据保存目录
        :param device: 数据存储设备 (推荐统一转为 cpu 数据保存)
        """
        self.save_dir = save_dir
        self.device = torch.device(device)
        self.data_registry: Dict[str, list] = {}  # 数据类型注册表
        os.makedirs(self.save_dir, exist_ok=True)  # 自动创建目录

    def add_tensor(self, tensor_name: str, tensor: torch.Tensor):
        """
        添加一条新数据
        :param tensor_name: 数据类型名称 (如 "accelerometer")
        :param tensor: 要保存的 tensor
        """
        # 数据预处理
        tensor = tensor.detach().to(self.device)
        
        # 注册新数据类型
        if tensor_name not in self.data_registry:
            self.data_registry[tensor_name] = []
        
        # 添加数据到缓存列表
        self.data_registry[tensor_name].append(tensor)

    def save_all(self, file_format: str = "npy"):
        """
        保存所有数据到本地文件
        :param file_format: 保存格式 (npy/pt)
        """
        for name, data_list in self.data_registry.items():
            if not data_list:
                continue
            
            # 合并数据为单个 tensor
            merged_tensor = torch.stack(data_list)  # shape: [num_samples, ...]
            
            # 构建文件路径
            file_path = os.path.join(self.save_dir, f"{name}.{file_format}")
            
            # 选择保存格式
            if file_format == "npy":
                np_data = merged_tensor.cpu().numpy()
                torch.save(np_data, file_path) if file_format == "pt" else np.save(file_path, np_data)
            elif file_format == "pt":
                torch.save(merged_tensor, file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}. Use 'npy' or 'pt'")
            
            print(f"Saved {len(data_list)} samples of '{name}' to {file_path}")

        # 清空缓存 (可选)
        self.data_registry.clear()