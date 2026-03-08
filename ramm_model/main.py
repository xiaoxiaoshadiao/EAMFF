import numpy as np
import torch
import torch.nn as nn
import os
# 导入自定义模块
from data_process_ramm import generate_fuel_cell_data_ramm, preprocess_data_ramm
from ramm_core import RAMM
from train_predict_ramm import train_model_ramm, predict_model_ramm, calculate_metrics_ramm
from visualization_ramm import (
    plot_train_history_ramm,
    plot_recovery_fitting_ramm,
    plot_long_step_predict_ramm,
    plot_metrics_radar_ramm
)

# 全局配置（严格对齐论文参数）
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 5000  # 长序列数据量（适配500窗口长度）
WINDOW_LEN = 500  # 输入序列长度
PREDICT_STEP = 100  # 预测步长（50/100/150可选）
M = 3  # 指数分解层数（拆分为4个子序列）
F_ENC = 128  # 编码维度
HIDDEN_DIM = 256  # HIIM全连接层隐藏维度
KERNEL_N = [2, 4, 6]  # 稳健分解核大小
EPOCHS = 5  # 训练轮数（适配长序列训练）
BATCH_SIZE = 16  # 批量大小

# 设置随机种子（保证可复现性）
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    print("===== RAMM模型端到端运行流程 =====")

    # 1. 数据生成与预处理（含电压恢复特征）
    print("\n【步骤1】生成并预处理燃料电池数据...")
    raw_data = generate_fuel_cell_data_ramm(n_samples=N_SAMPLES, seed=SEED)
    scaler, X_train, X_test, y_train, y_test = preprocess_data_ramm(
        raw_data, window_len=WINDOW_LEN, step=5, predict_step=PREDICT_STEP, seed=SEED
    )
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

    # 2. 构建DataLoader（批处理数据）
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化RAMM模型
    print("\n【步骤2】初始化RAMM模型...")
    in_features = X_train.shape[-1]
    ramm_model = RAMM(
        in_features=in_features,
        m=M,
        f_enc=F_ENC,
        hidden_dim=HIDDEN_DIM,
        kernel_n=KERNEL_N,
        predict_step=PREDICT_STEP
    ).to(DEVICE)
    # 打印模型结构
    print("RAMM模型结构：")
    print(ramm_model)

    # 4. 模型训练（含早停机制）
    print("\n【步骤3】训练模型...")
    trained_model, train_history = train_model_ramm(
        ramm_model, train_loader, val_loader, epochs=EPOCHS, lr=5e-5, patience=8, device=DEVICE
    )

    # 5. 模型预测（长步长）
    print("\n【步骤4】模型长步长预测...")
    y_pred = predict_model_ramm(trained_model, X_test, device=DEVICE)
    print(f"预测结果形状: {y_pred.shape}")

    # 6. 计算评估指标（重点关注电压）
    print("\n【步骤5】计算评估指标...")
    voltage_idx = raw_data.columns.get_loc("Voltage")
    metrics = calculate_metrics_ramm(y_test, y_pred, target_idx=voltage_idx)
    print("电压预测核心指标（长步长{}步）：".format(PREDICT_STEP))
    for k, v in metrics.items():
        if k == "MAPE":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.6f}")

    # 7. 结果可视化（生成图表文件）
    print("\n【步骤6】结果可视化...")
    plot_train_history_ramm(train_history)  # 训练损失/RMSE曲线
    plot_recovery_fitting_ramm(y_test, y_pred, scaler, voltage_idx)  # 恢复行为拟合图
    plot_long_step_predict_ramm(y_test, y_pred, scaler, voltage_idx, predict_step=PREDICT_STEP)  # 长步长对比图
    plot_metrics_radar_ramm(metrics)  # 评估指标雷达图

    # 清理临时模型文件（可选）
    if os.path.exists("ramm_best_model.pth"):
        os.remove("ramm_best_model.pth")

    print("\n===== RAMM模型端到端运行完成！=====")


if __name__ == "__main__":
    main()