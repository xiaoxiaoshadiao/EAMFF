import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['PingFang SC']  # macOS
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux
plt.rcParams['axes.unicode_minus'] = False


def plot_train_history_ramm(train_history, save_path="ramm_train_history.png"):
    """
    绘制训练损失和验证RMSE曲线
    :param train_history: 训练历史字典
    :param save_path: 保存路径
    """
    epochs = range(1, len(train_history['train_loss']) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 训练损失
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('训练损失 (MSE)', color='tab:red')
    ax1.plot(epochs, train_history['train_loss'], 'r-', label='训练损失')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 验证RMSE
    ax2 = ax1.twinx()
    ax2.set_ylabel('验证RMSE', color='tab:blue')
    ax2.plot(epochs, train_history['val_rmse'], 'b-', label='验证RMSE')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 标题和图例
    fig.suptitle('RAMM模型训练过程', fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练历史图已保存至: {save_path}")


def plot_recovery_fitting_ramm(y_true, y_pred, scaler, voltage_idx=0, save_path="ramm_recovery_fitting.png"):
    """
    绘制电压恢复行为拟合图
    :param y_true: 真实值（归一化后）
    :param y_pred: 预测值（归一化后）
    :param scaler: 归一化器
    :param voltage_idx: 电压特征索引
    :param save_path: 保存路径
    """

    # 反归一化
    def inverse_transform_voltage(data, scaler, voltage_idx):
        # 构造全零数组（匹配所有特征）
        data_full = np.zeros((data.shape[0], data.shape[1], scaler.n_features_in_))
        data_full[:, :, voltage_idx] = data
        # 反归一化
        data_inv = scaler.inverse_transform(data_full.reshape(-1, scaler.n_features_in_))
        # 提取电压列
        return data_inv[:, voltage_idx].reshape(data.shape[0], data.shape[1])

    # 反归一化电压
    y_true_voltage = inverse_transform_voltage(y_true[:, :, voltage_idx], scaler, voltage_idx)
    y_pred_voltage = inverse_transform_voltage(y_pred[:, :, voltage_idx], scaler, voltage_idx)

    # 选择第一个样本绘制
    sample_idx = 0
    steps = range(len(y_true_voltage[sample_idx]))

    plt.figure(figsize=(12, 6))
    plt.plot(steps, y_true_voltage[sample_idx], 'b-', label='真实电压', linewidth=1.5)
    plt.plot(steps, y_pred_voltage[sample_idx], 'r--', label='预测电压', linewidth=1.5)

    # 标注恢复区间（每100步一个恢复点，这里取前3个）
    for i in range(0, len(steps), 100):
        if i + 10 < len(steps):
            plt.axvspan(i, i + 10, alpha=0.2, color='green', label='恢复区间' if i == 0 else "")

    plt.xlabel('预测步长')
    plt.ylabel('电压 (V)')
    plt.title('RAMM模型电压恢复行为拟合效果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"电压恢复拟合图已保存至: {save_path}")


def plot_long_step_predict_ramm(y_true, y_pred, scaler, voltage_idx=0, predict_step=100,
                                save_path="ramm_long_step_predict.png"):
    """
    绘制长步长预测对比图
    :param y_true: 真实值（归一化后）
    :param y_pred: 预测值（归一化后）
    :param scaler: 归一化器
    :param voltage_idx: 电压特征索引
    :param predict_step: 预测步长
    :param save_path: 保存路径
    """

    # 反归一化电压
    def inverse_transform_voltage(data, scaler, voltage_idx):
        data_full = np.zeros((data.shape[0], data.shape[1], scaler.n_features_in_))
        data_full[:, :, voltage_idx] = data
        data_inv = scaler.inverse_transform(data_full.reshape(-1, scaler.n_features_in_))
        return data_inv[:, voltage_idx].reshape(data.shape[0], data.shape[1])

    y_true_voltage = inverse_transform_voltage(y_true[:, :, voltage_idx], scaler, voltage_idx)
    y_pred_voltage = inverse_transform_voltage(y_pred[:, :, voltage_idx], scaler, voltage_idx)

    # 计算不同步长的平均误差
    steps_range = [50, 100, 150] if predict_step >= 150 else [50, predict_step]
    step_errors = []
    for step in steps_range:
        if step > predict_step:
            continue
        # 前step步的平均误差
        error = np.mean(np.abs(y_true_voltage[:, :step] - y_pred_voltage[:, :step]))
        step_errors.append(error)

    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：第一个样本的长步长预测
    sample_idx = 0
    steps = range(predict_step)
    ax1.plot(steps, y_true_voltage[sample_idx], 'b-', label='真实电压', linewidth=1.5)
    ax1.plot(steps, y_pred_voltage[sample_idx], 'r--', label='预测电压', linewidth=1.5)
    ax1.set_xlabel('预测步长')
    ax1.set_ylabel('电压 (V)')
    ax1.set_title(f'RAMM模型{predict_step}步电压预测结果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：不同步长的平均误差
    ax2.bar([str(s) + '步' for s in steps_range], step_errors, color=['skyblue', 'orange', 'green'])
    ax2.set_xlabel('预测步长')
    ax2.set_ylabel('平均绝对误差 (V)')
    ax2.set_title('不同步长下的电压预测误差')
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(step_errors):
        ax2.text(i, v + 0.001, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"长步长预测图已保存至: {save_path}")


def plot_metrics_radar_ramm(metrics, save_path="ramm_metrics_radar.png"):
    """
    绘制评估指标雷达图
    :param metrics: 指标字典（MAE, MAPE, RMSE, R2）
    :param save_path: 保存路径
    """
    # 指标标准化（R²正向，其余负向）
    labels = ['MAE', 'MAPE', 'RMSE', 'R2']
    # 标准化：MAE/RMSE越小越好（除以最大值，取反），MAPE越小越好，R²越大越好
    max_mae = 0.1  # 经验最大值
    max_mape = 10  # 经验最大值
    max_rmse = 0.15  # 经验最大值

    values = [
        1 - min(metrics['MAE'] / max_mae, 1.0),  # MAE标准化
        1 - min(metrics['MAPE'] / max_mape, 1.0),  # MAPE标准化
        1 - min(metrics['RMSE'] / max_rmse, 1.0),  # RMSE标准化
        min(metrics['R2'], 1.0)  # R²标准化
    ]

    # 雷达图角度
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]  # 闭合
    angles += angles[:1]

    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='blue', label='RAMM模型')
    ax.fill(angles, values, alpha=0.25, color='blue')

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title('RAMM模型评估指标雷达图', size=14, pad=20)
    ax.grid(True)

    # 添加数值标注
    for angle, label, value in zip(angles[:-1], labels, values[:-1]):
        ax.text(angle, value + 0.05, f'{value:.2f}', ha='center', va='center')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"指标雷达图已保存至: {save_path}")


# 独立调试代码
if __name__ == "__main__":
    # 测试训练历史图
    print("测试训练历史图...")
    train_history = {
        'train_loss': [0.05, 0.04, 0.03, 0.025, 0.02],
        'val_rmse': [0.1, 0.09, 0.08, 0.075, 0.07]
    }
    plot_train_history_ramm(train_history)

    # 测试恢复拟合图
    print("\n测试恢复拟合图...")
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(np.random.randn(1000, 6))  # 模拟归一化器
    y_true = np.random.randn(10, 100, 6)
    y_pred = np.random.randn(10, 100, 6)
    plot_recovery_fitting_ramm(y_true, y_pred, scaler)

    # 测试长步长预测图
    print("\n测试长步长预测图...")
    plot_long_step_predict_ramm(y_true, y_pred, scaler, predict_step=100)

    # 测试指标雷达图
    print("\n测试指标雷达图...")
    metrics = {
        "MAE": 0.02,
        "MAPE": 5.0,
        "RMSE": 0.03,
        "R2": 0.88
    }
    plot_metrics_radar_ramm(metrics)
    print("可视化模块测试完成！")