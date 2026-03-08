import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_train_history(history):
    """
    绘制训练历史曲线（训练损失+验证RMSE）
    :param history: 训练历史字典 {"train_loss": [], "val_rmse": []}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 训练损失曲线
    ax1.plot(history["train_loss"], label="训练损失", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("FAMM模型训练损失变化")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 验证RMSE曲线
    ax2.plot(history["val_rmse"], label="验证RMSE", color="#ff7f0e")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.set_title("FAMM模型验证RMSE变化")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("famm_train_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_predict_vs_true(y_true, y_pred, scaler, feature_name="Voltage", sample_num=500, feature_names=None):
    """
    绘制预测值vs真实值曲线（整体+局部放大）
    :param y_true: 归一化后的真实值 (n_samples, predict_step, features)
    :param y_pred: 归一化后的预测值 (n_samples, predict_step, features)
    :param scaler: 数据归一化器（已fit过的MinMaxScaler）
    :param feature_name: 要可视化的特征（默认Voltage）
    :param sample_num: 局部放大的样本数
    :param feature_names: 特征名列表（容错用，优先用scaler的feature_names_in_）
    """
    # 反归一化（恢复原始尺度）
    n_features = y_true.shape[-1]
    y_true_reshaped = y_true.reshape(-1, n_features)
    y_pred_reshaped = y_pred.reshape(-1, n_features)

    y_true_inv = scaler.inverse_transform(y_true_reshaped)
    y_pred_inv = scaler.inverse_transform(y_pred_reshaped)

    # 找到特征索引（核心修复：增加容错逻辑）
    try:
        # 优先从scaler获取特征名
        feature_idx = scaler.feature_names_in_.tolist().index(feature_name)
    except AttributeError:
        # scaler无该属性时，用传入的feature_names
        if feature_names is not None:
            feature_idx = feature_names.index(feature_name)
        else:
            # 兜底：Voltage默认是最后一列
            feature_idx = -1
            print(f"警告：scaler无feature_names_in_属性，默认使用最后一列作为{feature_name}")

    # 提取数据（取前sample_num个点，防止样本数不足）
    sample_num = min(sample_num, len(y_true_inv))
    true_vis = y_true_inv[:sample_num, feature_idx]
    pred_vis = y_pred_inv[:sample_num, feature_idx]
    time_idx = np.arange(len(true_vis))

    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 整体对比
    ax1.plot(time_idx, true_vis, label="真实值", color="#2E86AB", linewidth=2)
    ax1.plot(time_idx, pred_vis, label="预测值", color="#F24236", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel(f"{feature_name} (V)")
    ax1.set_title(f"FAMM模型{feature_name}预测值vs真实值（整体）")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 局部放大（取后200个点，防止样本数不足）
    zoom_num = min(200, len(true_vis))
    local_true = true_vis[-zoom_num:]
    local_pred = pred_vis[-zoom_num:]
    local_time = time_idx[-zoom_num:]

    ax2.plot(local_time, local_true, label="真实值", color="#2E86AB", linewidth=2)
    ax2.plot(local_time, local_pred, label="预测值", color="#F24236", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("时间步")
    ax2.set_ylabel(f"{feature_name} (V)")
    ax2.set_title(f"FAMM模型{feature_name}预测值vs真实值（局部放大）")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"famm_{feature_name}_predict_vs_true.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_metrics_radar(metrics, model_name="FAMM"):
    """
    绘制评估指标雷达图（优化标准化逻辑，避免中心点聚集）
    :param metrics: 指标字典 {"MAE": xx, "MAPE": xx, "RMSE": xx, "R2": xx}
    :param model_name: 模型名称
    """
    # 定义指标类型（越小越好/越大越好）
    categories = ['MAE', 'MAPE', 'RMSE', 'R2']
    # 提取原始指标值
    mae = metrics['MAE']
    mape = metrics['MAPE']
    rmse = metrics['RMSE']
    r2 = metrics['R2']

    # ========== 核心优化：动态标准化 ==========
    # 1. 越小越好的指标（MAE/RMSE/MAPE）：非线性映射，避免数值趋近于1
    # MAE/RMSE映射：1 - (值 / (值 + 0.05)) （0.05为调节系数，可根据实际值调整）
    norm_mae = 1 - (mae / (mae + 0.05))
    norm_rmse = 1 - (rmse / (rmse + 0.05))
    # MAPE映射：1 - (mape / (mape + 5)) （MAPE是百分比，调节系数用5）
    norm_mape = 1 - (mape / (mape + 5))

    # 2. R2特殊处理：映射到[0,1]区间（解决负数问题）
    # 公式：(R2 + 2) / 3 → R2=-2时为0，R2=1时为1，覆盖常见R2范围
    norm_r2 = (r2 + 2) / 3
    norm_r2 = np.clip(norm_r2, 0, 1)  # 限制在0-1之间

    # 组合标准化后的值并闭合图形
    values = [norm_mae, norm_mape, norm_rmse, norm_r2]
    values += values[:1]  # 闭合图形

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合角度

    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    # 绘制指标曲线
    ax.plot(angles, values, 'o-', linewidth=2, color="#1f77b4", label=model_name)
    ax.fill(angles, values, alpha=0.25, color="#1f77b4")

    # ========== 优化显示效果 ==========
    # 设置刻度（增加细分刻度，增强区分度）
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    # 调整y轴范围和刻度，避免聚集在中心点
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    # 添加网格
    ax.grid(True, alpha=0.5)

    # 添加标题和图例
    ax.set_title(f"{model_name}模型评估指标雷达图", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=12)

    plt.tight_layout()
    plt.savefig(f"famm_{model_name}_metrics_radar.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印标准化后的值（便于调试）
    print(f"\n雷达图标准化后指标值：")
    print(f"  MAE (归一化): {norm_mae:.3f}")
    print(f"  MAPE (归一化): {norm_mape:.3f}")
    print(f"  RMSE (归一化): {norm_rmse:.3f}")
    print(f"  R2 (归一化): {norm_r2:.3f}")


# ==================== Debug测试 ====================
if __name__ == "__main__":
    print("=== 测试visualization.py功能 ===")

    # 1. 测试训练历史可视化
    print("\n1. 测试训练历史曲线绘制...")
    test_history = {
        "train_loss": [0.01, 0.008, 0.006, 0.005, 0.0045, 0.004, 0.0038, 0.0035, 0.0032, 0.003],
        "val_rmse": [0.009, 0.0085, 0.007, 0.0065, 0.006, 0.0058, 0.0055, 0.0052, 0.005, 0.0048]
    }
    plot_train_history(test_history)

    # 2. 测试预测值vs真实值绘制（模拟数据）
    print("\n2. 测试预测值vs真实值绘制...")
    from sklearn.preprocessing import MinMaxScaler

    # 模拟原始数据（先fit scaler，解决min_/scale_缺失问题）
    feature_names = ["Current", "Temp_anode", "Resistance", "Pressure_cathode_inlet", "Power", "Voltage"]
    n_samples_raw = 1000  # 模拟原始数据量
    raw_data = np.random.rand(n_samples_raw, len(feature_names))  # 模拟原始数据
    raw_data[:, -1] = 0.6 + 0.1 * raw_data[:, -1]  # Voltage特征范围调整到0.6~0.7V

    # 初始化并fit scaler（关键：必须先fit才能生成min_/scale_属性）
    scaler = MinMaxScaler()
    scaler.fit(raw_data)
    # 手动设置feature_names_in_（sklearn 1.0+版本支持）
    scaler.feature_names_in_ = np.array(feature_names)

    # 模拟预测/真实数据（归一化后的数据）
    n_samples = 500
    predict_step = 1
    n_features = len(feature_names)
    y_true = np.random.rand(n_samples, predict_step, n_features)
    y_pred = y_true + np.random.normal(0, 0.02, y_true.shape)  # 加小噪声模拟预测

    plot_predict_vs_true(y_true, y_pred, scaler, feature_name="Voltage", sample_num=500)

    # 3. 测试雷达图绘制（模拟小数值指标）
    print("\n3. 测试指标雷达图绘制...")
    test_metrics = {"MAE": 0.025, "MAPE": 91.93, "RMSE": 0.2116, "R2": -2.4223}  # 模拟你的真实指标
    plot_metrics_radar(test_metrics, model_name="FAMM")

    print("\n=== visualization.py测试完成，可视化功能正常 ===")