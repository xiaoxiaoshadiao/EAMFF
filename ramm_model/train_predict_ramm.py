import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def train_model_ramm(model, train_loader, val_loader, epochs=50, lr=5e-5, patience=8, device="cuda"):
    """
    训练RAMM模型（含早停机制）
    :param model: RAMM模型实例
    :param train_loader: 训练DataLoader
    :param val_loader: 验证DataLoader
    :param epochs: 最大训练轮数
    :param lr: 学习率
    :param patience: 早停耐心值
    :param device: 训练设备
    :return: 训练好的模型, 训练历史（loss/val_rmse）
    """
    # 优化器（AdamW，论文推荐）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # 损失函数（MSE + L1正则化）
    criterion = nn.MSELoss()
    l1_reg = 1e-6  # L1正则化系数

    # 早停相关
    best_val_rmse = float('inf')
    patience_counter = 0
    best_model_weights = None

    # 训练历史
    train_history = {
        'train_loss': [],
        'val_rmse': []
    }

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.float32)

            # 前向传播
            pred = model(batch_x)
            # 计算损失（MSE + L1正则化）
            loss = criterion(pred, batch_y)
            # L1正则化
            l1_loss = sum(p.abs().sum() for p in model.parameters()) * l1_reg
            loss += l1_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_epoch += loss.item() * batch_x.size(0)

        # 计算epoch平均训练损失
        train_loss_avg = train_loss_epoch / len(train_loader.dataset)
        train_history['train_loss'].append(train_loss_avg)

        # 验证阶段
        model.eval()
        val_rmse_epoch = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device, dtype=torch.float32)

                pred = model(batch_x)
                mse = criterion(pred, batch_y)
                rmse = torch.sqrt(mse)
                val_rmse_epoch += rmse.item() * batch_x.size(0)

        # 计算epoch平均验证RMSE
        val_rmse_avg = val_rmse_epoch / len(val_loader.dataset)
        train_history['val_rmse'].append(val_rmse_avg)

        # 打印日志
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss_avg:.6f}, Val RMSE: {val_rmse_avg:.6f}")

        # 早停判断
        if val_rmse_avg < best_val_rmse:
            best_val_rmse = val_rmse_avg
            best_model_weights = model.state_dict()
            patience_counter = 0
            # 保存最佳模型
            torch.save(best_model_weights, "ramm_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发！最佳验证RMSE: {best_val_rmse:.6f}")
                break

    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    return model, train_history


def predict_model_ramm(model, X_test, device="cuda"):
    """
    RAMM模型预测
    :param model: 训练好的RAMM模型
    :param X_test: 测试集输入 (n_samples, window_len, in_features)
    :param device: 预测设备
    :return: 预测结果 (n_samples, predict_step, in_features)
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(X_test_tensor)
    # 转换为numpy数组
    pred_np = pred.cpu().numpy()
    return pred_np


def calculate_metrics_ramm(y_true, y_pred, target_idx=0):
    """
    计算评估指标（MAE, MAPE, RMSE, R²）
    :param y_true: 真实值 (n_samples, predict_step, in_features)
    :param y_pred: 预测值 (n_samples, predict_step, in_features)
    :param target_idx: 目标特征索引（默认0=Voltage）
    :return: 指标字典
    """
    # 提取目标特征（电压）
    y_true_target = y_true[:, :, target_idx].reshape(-1)
    y_pred_target = y_pred[:, :, target_idx].reshape(-1)

    # 避免除以0（MAPE计算）
    mask = y_true_target != 0
    y_true_target = y_true_target[mask]
    y_pred_target = y_pred_target[mask]

    # MAE
    mae = np.mean(np.abs(y_true_target - y_pred_target))
    # MAPE
    mape = np.mean(np.abs((y_true_target - y_pred_target) / y_true_target)) * 100
    # RMSE
    rmse = np.sqrt(np.mean((y_true_target - y_pred_target) ** 2))
    # R²
    ss_total = np.sum((y_true_target - np.mean(y_true_target)) ** 2)
    ss_res = np.sum((y_true_target - y_pred_target) ** 2)
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0.0

    return {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2
    }


# 独立调试代码
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模拟数据
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = 16
    X_train = torch.randn(100, 500, 6)
    y_train = torch.randn(100, 100, 6)
    X_val = torch.randn(20, 500, 6)
    y_val = torch.randn(20, 100, 6)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    from ramm_core import RAMM

    model = RAMM(in_features=6, predict_step=100).to(device)

    # 测试训练
    print("测试模型训练...")
    trained_model, train_history = train_model_ramm(
        model, train_loader, val_loader, epochs=2, lr=5e-5, patience=2, device=device
    )
    print(f"训练历史 - 训练损失: {train_history['train_loss']}, 验证RMSE: {train_history['val_rmse']}")

    # 测试预测
    print("\n测试模型预测...")
    X_test = np.random.randn(10, 500, 6)
    y_pred = predict_model_ramm(trained_model, X_test, device=device)
    print(f"预测结果形状: {y_pred.shape}")

    # 测试指标计算
    print("\n测试指标计算...")
    y_true = np.random.randn(10, 100, 6)
    metrics = calculate_metrics_ramm(y_true, y_pred, target_idx=0)
    print("评估指标：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("训练预测模块测试完成！")