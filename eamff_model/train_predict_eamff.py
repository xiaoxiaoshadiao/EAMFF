import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 定义损失函数（对齐论文6.1.4节）
class EAMFFLoss(nn.Module):
    def __init__(self, lambda_cons=0.1):
        super(EAMFFLoss, self).__init__()
        self.lambda_cons = lambda_cons
        self.mse = nn.MSELoss()

    def forward(self, X_hat, y_true, Xs, Xbg, Xr, predict_step):
        """
        :param X_hat: 预测值 (batch, O, F)
        :param y_true: 真实值 (batch, O, F)
        :param Xs/Xbg/Xr: 三路专家分量 (batch, O, F)
        :param predict_step: 预测步长O
        :return: total_loss, pred_loss, cons_loss
        """
        # 1. 预测损失（公式6-8）
        pred_loss = self.mse(X_hat, y_true)

        # 2. 分量一致性损失（公式6-9）- 仅中期步长启用
        if 10 < predict_step < 100:
            # Frobenius范数的平方 = 所有元素的平方和
            cons_loss = (
                                torch.sum((Xs - Xbg) ** 2) +
                                torch.sum((Xs - Xr) ** 2) +
                                torch.sum((Xbg - Xr) ** 2)
                        ) / (3 * Xs.numel())  # numel()=总元素数
        else:
            cons_loss = torch.tensor(0.0, device=X_hat.device)

        # 3. 总损失（公式6-10）
        total_loss = pred_loss + self.lambda_cons * cons_loss

        return total_loss, pred_loss, cons_loss


def train_model_eamff(model, train_loader, val_loader, epochs, lr, patience, lambda_cons, device):
    """
    端到端训练EAMFF模型
    :param model: EAMFF模型
    :param train_loader: 训练DataLoader
    :param val_loader: 验证DataLoader
    :param epochs: 训练轮数
    :param lr: 学习率
    :param patience: 早停耐心值
    :param lambda_cons: 一致性损失权重
    :param device: 设备
    :return: trained_model, train_history
    """
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # 损失函数
    criterion = EAMFFLoss(lambda_cons=lambda_cons)

    # 训练历史
    train_history = {
        "train_loss": [], "val_loss": [],
        "train_pred_loss": [], "val_pred_loss": [],
        "train_cons_loss": [], "val_cons_loss": []
    }

    # 早停初始化
    best_val_loss = float("inf")
    patience_counter = 0

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0.0
        train_pred_loss_epoch = 0.0
        train_cons_loss_epoch = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for x_batch, o_batch, y_batch in pbar:
            x_batch = x_batch.to(device).float()
            o_batch = o_batch.to(device).float().unsqueeze(1)  # (batch,1)
            y_batch = y_batch.to(device).float()

            # 取第一个样本的步长作为当前batch的步长（简化版，实际可支持多步长）
            predict_step = int(o_batch[0, 0].item())

            # 前向传播
            X_hat, alpha, Xs, Xbg, Xr = model(x_batch, predict_step)

            # 计算损失
            total_loss, pred_loss, cons_loss = criterion(X_hat, y_batch, Xs, Xbg, Xr, predict_step)

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 累计损失
            train_loss_epoch += total_loss.item()
            train_pred_loss_epoch += pred_loss.item()
            train_cons_loss_epoch += cons_loss.item()

            pbar.set_postfix({"loss": total_loss.item()})

        # 验证阶段
        model.eval()
        val_loss_epoch = 0.0
        val_pred_loss_epoch = 0.0
        val_cons_loss_epoch = 0.0

        with torch.no_grad():
            for x_batch, o_batch, y_batch in val_loader:
                x_batch = x_batch.to(device).float()
                o_batch = o_batch.to(device).float().unsqueeze(1)
                y_batch = y_batch.to(device).float()

                predict_step = int(o_batch[0, 0].item())
                X_hat, alpha, Xs, Xbg, Xr = model(x_batch, predict_step)
                total_loss, pred_loss, cons_loss = criterion(X_hat, y_batch, Xs, Xbg, Xr, predict_step)

                val_loss_epoch += total_loss.item()
                val_pred_loss_epoch += pred_loss.item()
                val_cons_loss_epoch += cons_loss.item()

        # 平均损失
        train_loss_avg = train_loss_epoch / len(train_loader)
        train_pred_loss_avg = train_pred_loss_epoch / len(train_loader)
        train_cons_loss_avg = train_cons_loss_epoch / len(train_loader)

        val_loss_avg = val_loss_epoch / len(val_loader)
        val_pred_loss_avg = val_pred_loss_epoch / len(val_loader)
        val_cons_loss_avg = val_cons_loss_epoch / len(val_loader)

        # 记录历史
        train_history["train_loss"].append(train_loss_avg)
        train_history["val_loss"].append(val_loss_avg)
        train_history["train_pred_loss"].append(train_pred_loss_avg)
        train_history["val_pred_loss"].append(val_pred_loss_avg)
        train_history["train_cons_loss"].append(train_cons_loss_avg)
        train_history["val_cons_loss"].append(val_cons_loss_avg)

        # 打印日志
        print(f"\nEpoch {epoch + 1}:")
        print(f"  训练总损失: {train_loss_avg:.6f}, 验证总损失: {val_loss_avg:.6f}")
        print(f"  训练预测损失: {train_pred_loss_avg:.6f}, 验证预测损失: {val_pred_loss_avg:.6f}")
        print(f"  训练一致性损失: {train_cons_loss_avg:.6f}, 验证一致性损失: {val_cons_loss_avg:.6f}")

        # 早停判断
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "eamff_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发！最佳验证损失: {best_val_loss:.6f}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load("eamff_best_model.pth"))
    return model, train_history


def fine_tune_model_eamff(model, train_loader, val_loader, epochs, lr, patience, lambda_cons, device):
    """
    迁移学习微调：冻结主干，仅微调门控和输出层
    :param model: 预训练EAMFF模型
    :return: fine_tuned_model, train_history
    """
    # 冻结主干（专家封装模块）
    for param in model.expert_encap.parameters():
        param.requires_grad = False

    # 仅微调门控和融合模块
    optimizer = optim.AdamW(
        list(model.gating_routing.parameters()) + list(model.fusion_pred.parameters()),
        lr=lr, weight_decay=1e-5
    )
    criterion = EAMFFLoss(lambda_cons=lambda_cons)

    # 训练历史（简化版）
    train_history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    # 微调循环
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for x_batch, o_batch, y_batch in tqdm(train_loader, desc=f"微调Epoch {epoch + 1}/{epochs}"):
            x_batch = x_batch.to(device).float()
            o_batch = o_batch.to(device).float().unsqueeze(1)
            y_batch = y_batch.to(device).float()

            predict_step = int(o_batch[0, 0].item())
            X_hat, alpha, Xs, Xbg, Xr = model(x_batch, predict_step)
            total_loss, _, _ = criterion(X_hat, y_batch, Xs, Xbg, Xr, predict_step)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, o_batch, y_batch in val_loader:
                x_batch = x_batch.to(device).float()
                o_batch = o_batch.to(device).float().unsqueeze(1)
                y_batch = y_batch.to(device).float()

                predict_step = int(o_batch[0, 0].item())
                X_hat, alpha, Xs, Xbg, Xr = model(x_batch, predict_step)
                total_loss, _, _ = criterion(X_hat, y_batch, Xs, Xbg, Xr, predict_step)

                val_loss += total_loss.item()

        # 平均损失
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        train_history["train_loss"].append(train_loss_avg)
        train_history["val_loss"].append(val_loss_avg)

        print(f"微调Epoch {epoch + 1}: 训练损失={train_loss_avg:.6f}, 验证损失={val_loss_avg:.6f}")

        # 早停
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), "eamff_finetuned_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"微调早停触发！最佳验证损失: {best_val_loss:.6f}")
                break

    # 加载微调后模型
    model.load_state_dict(torch.load("eamff_finetuned_model.pth"))
    return model, train_history


def predict_model_eamff(model, X_test, predict_step, device):
    """
    模型预测
    :param model: 训练好的EAMFF模型
    :param X_test: 测试输入 (n_samples, window_len, in_features)
    :param predict_step: 预测步长O
    :param device: 设备
    :return: y_pred (n_samples, O, in_features)
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_tensor, _, _, _, _ = model(X_test_tensor, predict_step)

    y_pred = y_pred_tensor.cpu().numpy()
    return y_pred


def calculate_metrics_eamff(y_true, y_pred, target_idx=0):
    """
    计算评估指标（重点关注Voltage）
    :param y_true: 真实值 (n_samples, O, F)
    :param y_pred: 预测值 (n_samples, O, F)
    :param target_idx: 目标特征索引（Voltage）
    :return: metrics字典
    """
    # 展平数据（n_samples*O, 1）
    y_true_flat = y_true[:, :, target_idx].flatten()
    y_pred_flat = y_pred[:, :, target_idx].flatten()

    # 计算指标
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100  # 避免除0
    r2 = r2_score(y_true_flat, y_pred_flat)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }
    return metrics


# 独立调试代码
if __name__ == "__main__":
    print("=== 调试train_predict_eamff.py ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 测试损失函数
    print("\n1. 测试损失函数：")
    criterion = EAMFFLoss(lambda_cons=0.1)
    X_hat = torch.randn(4, 10, 6).to(device)
    y_true = torch.randn(4, 10, 6).to(device)
    Xs = torch.randn(4, 10, 6).to(device)
    Xbg = torch.randn(4, 10, 6).to(device)
    Xr = torch.randn(4, 10, 6).to(device)

    # 中期步长（O=50）
    total_loss, pred_loss, cons_loss = criterion(X_hat, y_true, Xs, Xbg, Xr, 50)
    print(f"中期步长(O=50) - 总损失: {total_loss:.4f}, 预测损失: {pred_loss:.4f}, 一致性损失: {cons_loss:.4f}")

    # 短期步长（O=5）
    total_loss, pred_loss, cons_loss = criterion(X_hat, y_true, Xs, Xbg, Xr, 5)
    print(f"短期步长(O=5) - 总损失: {total_loss:.4f}, 预测损失: {pred_loss:.4f}, 一致性损失: {cons_loss:.4f}")

    # 2. 测试指标计算
    print("\n2. 测试指标计算：")
    y_true = np.random.randn(100, 10, 6)
    y_pred = np.random.randn(100, 10, 6)
    metrics = calculate_metrics_eamff(y_true, y_pred, target_idx=0)
    print(
        f"指标结果: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%, R2={metrics['R2']:.4f}")

    print("✅ 训练预测模块调试通过")