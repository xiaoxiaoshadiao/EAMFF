import numpy as np
import torch
import torch.nn as nn
import os
# 导入自定义模块
from data_process_eamff import generate_fuel_cell_data_eamff, preprocess_data_eamff
from eamff_core import EAMFF
from train_predict_eamff import (
    train_model_eamff, predict_model_eamff, calculate_metrics_eamff,
    fine_tune_model_eamff
)
from visualization_eamff import (
    plot_train_history_eamff,
    plot_multi_step_predict_eamff,
    plot_gating_weights_eamff,
    plot_component_consistency_eamff,
    plot_metrics_radar_eamff
)

# 全局配置（严格对齐论文参数）
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 8000  # 多步长训练需更多数据
WINDOW_LEN = 500  # 输入序列长度
SUPPORTED_O = [10, 50, 100, 150]  # 支持的预测步长（覆盖短/中/长）
FAMM_D_MODEL = 64  # FAMM编码维度
RAMM_F_ENC = 128  # RAMM编码维度
GATING_HIDDEN_DIM = 64  # 门控权重函数隐藏维度
LAMBDA_CONS = 0.1  # 一致性损失权重
EPOCHS = 5  # 训练轮数
BATCH_SIZE = 16  # 批量大小
WORK_MODE = "inference"  # 工作模式："inference"（端到端）或 "transfer"（迁移学习）

# 设置随机种子（保证可复现性）
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    print(f"===== EAMFF模型端到端运行流程（{WORK_MODE}模式）=====")

    # 1. 数据生成与预处理（多步长样本）
    print("\n【步骤1】生成并预处理多步长燃料电池数据...")
    raw_data = generate_fuel_cell_data_eamff(n_samples=N_SAMPLES, seed=SEED)
    # 生成所有支持步长的训练/测试集
    datasets = {}
    for o in SUPPORTED_O:
        scaler, X_train, X_test, y_train, y_test = preprocess_data_eamff(
            raw_data, window_len=WINDOW_LEN, step=5, predict_step=o, seed=SEED
        )
        datasets[o] = (scaler, X_train, X_test, y_train, y_test)
    # 取最后一个步长的scaler和数据用于后续演示（实际训练包含所有步长）
    scaler = datasets[SUPPORTED_O[-1]][0]
    X_train = datasets[SUPPORTED_O[-1]][1]
    X_test = datasets[SUPPORTED_O[-1]][2]
    y_train = datasets[SUPPORTED_O[-1]][3]
    y_test = datasets[SUPPORTED_O[-1]][4]
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

    # 2. 构建DataLoader（含步长标签）
    from torch.utils.data import DataLoader, TensorDataset
    # 生成步长标签（每个样本的目标步长，这里统一用最大步长，实际训练为混合步长）
    o_labels_train = np.full(len(X_train), SUPPORTED_O[-1])
    o_labels_test = np.full(len(X_test), SUPPORTED_O[-1])
    # 构建数据集（输入序列 + 步长标签 + 真实值）
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),  # 指定数据类型
        torch.tensor(o_labels_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(o_labels_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化EAMFF模型
    print("\n【步骤2】初始化EAMFF模型...")
    in_features = X_train.shape[-1]
    eamff_model = EAMFF(
        in_features=in_features,
        famm_d_model=FAMM_D_MODEL,
        ramm_f_enc=RAMM_F_ENC,
        gating_hidden_dim=GATING_HIDDEN_DIM,
        max_o=max(SUPPORTED_O)
    ).to(DEVICE)
    print("EAMFF模型结构：")
    print(eamff_model)

    # 4. 模型训练/迁移微调
    print("\n【步骤3】模型训练/迁移微调...")
    if WORK_MODE == "inference":
        # 端到端训练（覆盖所有步长）
        trained_model, train_history = train_model_eamff(
            eamff_model, train_loader, val_loader,
            epochs=EPOCHS, lr=3e-5, patience=10,
            lambda_cons=LAMBDA_CONS, device=DEVICE
        )
    else:
        # 迁移学习：冻结主干，微调门控与输出层
        trained_model, train_history = fine_tune_model_eamff(
            eamff_model, train_loader, val_loader,
            epochs=30, lr=1e-5, patience=5,
            lambda_cons=LAMBDA_CONS, device=DEVICE
        )

    # ========== 新增：打印train_history的键名，方便调试 ==========
    print(f"\n训练历史可用键: {list(train_history.keys())}")

    # 5. 多步长预测验证
    print("\n【步骤4】多步长模型预测...")
    all_metrics = {}
    voltage_idx = raw_data.drop("Time", axis=1).columns.tolist().index("Voltage")
    for o in SUPPORTED_O:
        X_test_o = datasets[o][2]
        y_test_o = datasets[o][4]
        y_pred_o = predict_model_eamff(trained_model, X_test_o, predict_step=o, device=DEVICE)
        # 计算指标（重点关注电压）
        metrics_o = calculate_metrics_eamff(y_test_o, y_pred_o, target_idx=voltage_idx)
        all_metrics[o] = metrics_o
        print(f"\n电压预测指标（步长{o}步）：")
        for k, v in metrics_o.items():
            if k == "MAPE":
                print(f"  {k}: {v:.2f}%")
            elif k == "能量守恒误差":
                print(f"  {k}: {v:.2f}%")
            else:
                print(f"  {k}: {v:.6f}")

    # 6. 结果可视化
    print("\n【步骤5】结果可视化...")
    # 安全调用：即使train_history不完整也不会崩溃
    plot_train_history_eamff(train_history)  # 训练损失曲线

    # ========== 核心修改：构造适配可视化函数的数据集（仅保留4个元素） ==========
    # 解决 "too many values to unpack (expected 4)" 错误
    vis_datasets = {}
    for o in SUPPORTED_O:
        # 构造4个元素（前两个占位，后两个为X_test/y_test），适配可视化函数的解包逻辑
        vis_datasets[o] = (None, None, datasets[o][2], datasets[o][4])

    # 多步长预测对比（使用适配后的vis_datasets）
    plot_multi_step_predict_eamff(vis_datasets, trained_model, scaler, voltage_idx, device=DEVICE)

    # 门控权重分析（绘制权重随步长变化）
    plot_gating_weights_eamff(trained_model, max_o=max(SUPPORTED_O), device=DEVICE)
    # 分量一致性分析
    plot_component_consistency_eamff(trained_model, X_test[:10], predict_step=50, device=DEVICE)
    # 指标雷达图（以O=100步为例）
    if 100 in all_metrics:
        plot_metrics_radar_eamff(all_metrics[100], model_name="EAMFF")
    else:
        print("警告：步长100无指标数据，跳过雷达图绘制")

    # 清理临时模型文件（可选）
    for f in ["eamff_best_model.pth", "eamff_finetuned_model.pth"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"已清理临时文件: {f}")

    print("\n===== EAMFF模型端到端运行完成！=====")


if __name__ == "__main__":
    main()