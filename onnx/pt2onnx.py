import torch
import torch.onnx
model_path = "/home/huang/wheel_leg/wheel_legged_genesis_new/locomotion/logs/Noise_1_eval/2026-01-15_22-45-34_ckpt150/policy.pt"
model = torch.jit.load(model_path)
model.eval()
dummy_input = torch.randn(1, 186)  # ONNX 需要通过一次前向传播来追踪模型结构，因此需要虚拟输入,必须与模型实际输入形状一致（如(batch_size=1, features=174)）
onnx_model_path = '/home/huang/wheel_leg/wheel_legged_genesis_new/onnx/model_1.onnx'
# 导出为 ONNX 格式
torch.onnx.export(
    model,                   # 待导出的模型
    dummy_input,             # 模型输入（用于确定输入/输出形状）
    onnx_model_path,         # 输出路径
    input_names=['input_186'],   # 输入张量的名称
    output_names=['output_3'], # 输出张量的名称
    opset_version=11         # ONNX操作集版本（决定支持哪些算子）
)
print(f"模型已成功导出为 {onnx_model_path}")
