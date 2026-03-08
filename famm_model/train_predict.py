import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")

# 导入自定义模块
from data_process import generate_fuel_cell_data, preprocess_data
from famm_core import FAMM

# 设置全局随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置（优先GPU，无则CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-4, weight_decay=1e-5, patience=5):
    """
    模型训练流程（含早停机制）
    :param model: FAMM模型
    :param train_loader: 训练集DataLoader
    :param val_loader: 验证集DataLoader
    :param epochs: 最大训练轮数
    :param lr: 学习率
    :param weight_decay: 权重衰减
    :param patience: 早停耐心值
    :return: 训练好的模型 + 训练历史
    """
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()  # 论文指定MSE损失

    # 早停相关
    best_val_rmse = float('inf')
    patience_counter = 0
    train_history = {"train_loss": [], "val_rmse": []}

    # 训练循环
    model.train()  # 初始化为训练模式
    for epoch in range(epochs):
        # 训练阶段
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            # 前向传播
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_history["train_loss"].append(avg_train_loss)

        # 验证阶段（关键：验证后必须切回train模式）
        val_rmse = evaluate_model(model, val_loader, criterion)
        train_history["val_rmse"].append(val_rmse)
        model.train()  # 验证完成后切回训练模式！！！

        # 打印日志
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val RMSE: {val_rmse:.6f}")

        # 早停判断
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            # 保存最优模型（兼容CPU/GPU）
            torch.save({
                'model_state_dict': model.state_dict(),
                'device': str(device)
            }, "famm_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发！最优验证集RMSE: {best_val_rmse:.6f}")
                break

    # 加载最优模型
    checkpoint = torch.load("famm_best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_history


def evaluate_model(model, data_loader, criterion):
    """
    模型评估（计算RMSE）
    :param model: FAMM模型
    :param data_loader: 验证/测试集DataLoader
    :param criterion: 损失函数
    :return: RMSE
    """
    model.eval()  # 临时切换为评估模式
    val_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    avg_val_loss = val_loss / len(data_loader.dataset)
    val_rmse = np.sqrt(avg_val_loss)
    return val_rmse


def predict_model(model, X_test):
    """
    模型预测
    :param model: 训练好的FAMM模型
    :param X_test: 测试集输入 (n_samples, window_len, features)
    :return: 预测结果 (n_samples, predict_step, features)
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_test_tensor)
    return y_pred.cpu().numpy()


def calculate_metrics(y_true, y_pred, feature_idx):
    """
    计算评估指标（MAE、MAPE、RMSE、R²），重点关注电压特征
    :param y_true: 真实值 (n_samples, predict_step, features)
    :param y_pred: 预测值 (n_samples, predict_step, features)
    :param feature_idx: 电压特征的索引
    :return: 指标字典
    """
    # 提取电压维度
    y_true_voltage = y_true[:, :, feature_idx].reshape(-1)
    y_pred_voltage = y_pred[:, :, feature_idx].reshape(-1)

    # 计算指标
    mae = mean_absolute_error(y_true_voltage, y_pred_voltage)
    rmse = np.sqrt(mean_squared_error(y_true_voltage, y_pred_voltage))
    r2 = r2_score(y_true_voltage, y_pred_voltage)
    # MAPE（避免除零）
    mape = np.mean(np.abs((y_true_voltage - y_pred_voltage) / (y_true_voltage + 1e-8))) * 100

    metrics = {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2
    }
    return metrics


# ==================== Debug测试 ====================
if __name__ == "__main__":
    print("=== 测试train_predict.py功能 ===")

    # 1. 数据准备
    print("\n1. 生成并预处理数据...")
    raw_data = generate_fuel_cell_data(n_samples=2000)
    scaler, X_train, X_test, y_train, y_test = preprocess_data(
        raw_data, window_len=300, step=10, predict_step=1
    )
    print(f"训练集输入形状: {X_train.shape}, 训练集输出形状: {y_train.shape}")
    print(f"测试集输入形状: {X_test.shape}, 测试集输出形状: {y_test.shape}")

    # 2. 构建DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. 初始化模型
    print("\n2. 初始化FAMM模型...")
    in_features = X_train.shape[-1]
    famm_model = FAMM(in_features=in_features, d_model=64, predict_step=1).to(device)

    # 4. 训练模型（少量轮数测试）
    print("\n3. 训练模型（测试用，仅训练10轮）...")
    trained_model, train_history = train_model(
        famm_model, train_loader, val_loader, epochs=10, lr=1e-4, patience=3
    )

    # 5. 模型预测
    print("\n4. 模型预测...")
    y_pred = predict_model(trained_model, X_test)
    print(f"预测结果形状: {y_pred.shape}")

    # 6. 计算评估指标（电压特征索引：raw_data.columns.get_loc("Voltage")）
    print("\n5. 计算评估指标（重点关注电压）...")
    voltage_idx = raw_data.columns.get_loc("Voltage")
    metrics = calculate_metrics(y_test, y_pred, voltage_idx)
    print("电压预测指标：")
    for k, v in metrics.items():
        if k == "MAPE":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.6f}")

    # 清理生成的模型文件（可选）
    if os.path.exists("famm_best_model.pth"):
        os.remove("famm_best_model.pth")

    print("\n=== train_predict.py测试完成，训练/预测/评估功能正常 ===")