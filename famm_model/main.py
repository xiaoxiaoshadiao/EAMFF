import numpy as np
import torch
import os
# 导入自定义模块
from data_process import generate_fuel_cell_data, preprocess_data
from famm_core import FAMM
from train_predict import train_model, evaluate_model, predict_model, calculate_metrics
from visualization import plot_train_history, plot_predict_vs_true, plot_metrics_radar

# 全局配置
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 2000  # 数据量
WINDOW_LEN = 300  # 输入窗口长度
PREDICT_STEP = 1  # 预测步长
EPOCHS = 10  # 训练轮数
BATCH_SIZE = 32  # 批次大小

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    print("===== FAMM模型端到端运行流程 =====")

    # 1. 数据生成与预处理
    print("\n【步骤1】生成并预处理燃料电池数据...")
    raw_data = generate_fuel_cell_data(n_samples=N_SAMPLES)
    scaler, X_train, X_test, y_train, y_test = preprocess_data(
        raw_data, window_len=WINDOW_LEN, step=10, predict_step=PREDICT_STEP
    )
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

    # 2. 构建DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化模型
    print("\n【步骤2】初始化FAMM模型...")
    in_features = X_train.shape[-1]
    famm_model = FAMM(in_features=in_features, d_model=64, predict_step=PREDICT_STEP).to(DEVICE)

    # 4. 训练模型
    print("\n【步骤3】训练模型...")
    trained_model, train_history = train_model(
        famm_model, train_loader, val_loader, epochs=EPOCHS, lr=1e-4, patience=3
    )

    # 5. 模型预测
    print("\n【步骤4】模型预测...")
    y_pred = predict_model(trained_model, X_test)
    print(f"预测结果形状: {y_pred.shape}")

    # 6. 计算评估指标
    print("\n【步骤5】计算评估指标...")
    voltage_idx = raw_data.columns.get_loc("Voltage")
    metrics = calculate_metrics(y_test, y_pred, voltage_idx)
    print("电压预测指标：")
    for k, v in metrics.items():
        if k == "MAPE":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.6f}")


    # 7. 结果可视化
    print("\n【步骤6】结果可视化...")
    plot_train_history(train_history)  # 训练历史
    # 新增feature_names参数，兜底容错
    plot_predict_vs_true(y_test, y_pred, scaler, feature_name="Voltage",
                         feature_names=raw_data.columns.tolist())  # 预测对比
    plot_metrics_radar(metrics)  # 指标雷达图

    # 清理临时文件
    if os.path.exists("famm_best_model.pth"):
        os.remove("famm_best_model.pth")

    print("\n===== FAMM模型运行完成！=====")


if __name__ == "__main__":
    main()