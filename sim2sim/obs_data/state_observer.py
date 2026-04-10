import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 配置参数
class Config:
    input_dim = 32    # 两帧合并后的维度 (16*2)
    hidden_dim = 128
    output_dim = 3
    batch_size = 64
    learning_rate = 0.001
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理函数
def prepare_data(obs, targets):
    # 创建两帧数据集
    X, y = [], []
    for i in range(1, len(obs)):
        X.append(np.concatenate([obs[i-1], obs[i]]))
        y.append(targets[i])
    return np.array(X), np.array(y)

# 定义MLP模型
class VelocityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.input_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.hidden_dim, Config.hidden_dim//2),
            nn.Tanh(),
            nn.Linear(Config.hidden_dim//2, Config.output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 训练函数
def train_model(model, X_train, y_train, X_test, y_test):
    # 转换为PyTorch张量
    train_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=Config.batch_size, 
        shuffle=True
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(Config.device)
            y_batch = y_batch.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 验证评估
        model.eval()
        with torch.no_grad():
            test_inputs = torch.FloatTensor(X_test).to(Config.device)
            test_targets = torch.FloatTensor(y_test).to(Config.device)
            val_loss = criterion(model(test_inputs), test_targets)
            
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_mlp_model.pth")
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{Config.epochs} | "
                  f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f}")
    
    return best_loss

# 主程序
if __name__ == "__main__":
    # 加载数据
    base_ang_vel = np.load("base_ang_vel.npy")
    base_lin_vel = np.load("base_lin_vel.npy")
    dof_pos = np.load("dof_pos.npy")
    dof_vel = np.load("dof_vel.npy")
    projected_gravity = np.load("projected_gravity.npy")
    
    # 合并观测数据
    obs = np.concatenate([base_ang_vel, dof_pos, dof_vel, projected_gravity], axis=1)
    targets = base_lin_vel
    
    # 创建两帧数据集
    X, y = prepare_data(obs, targets)
    print(f"原始数据形状: X{X.shape}, y{y.shape}")
    
    # 数据标准化
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, 
        test_size=0.2, 
        shuffle=False  # 时间序列数据不shuffle
    )
    
    # 初始化模型
    model = VelocityPredictor().to(Config.device)
    print("模型结构:\n", model)
    
    # 开始训练
    print("\n开始训练...")
    final_loss = train_model(model, X_train, y_train, X_test, y_test)
    print(f"\n最佳验证损失: {final_loss:.4f}")
    
    # 保存JIT模型
    model.load_state_dict(torch.load("best_mlp_model.pth"))
    example_input = torch.randn(1, Config.input_dim).to(Config.device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("velocity_predictor_jit.pt")
    print("\nJIT模型已保存为 velocity_predictor_jit.pt")
    
    # 测试推理
    with torch.no_grad():
        test_input = torch.FloatTensor(X_test[:1]).to(Config.device)
        prediction = traced_model(test_input)
        print(f"\n测试输入形状: {test_input.shape}")
        print(f"JIT模型输出示例: {prediction.cpu().numpy()}")
        print(f"真实值: {y_scaler.inverse_transform(y_test[:1])[0]}")
        print(f"预测值: {y_scaler.inverse_transform(prediction.cpu().numpy())[0]}")