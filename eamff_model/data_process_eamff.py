import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def generate_fuel_cell_data_eamff(n_samples=8000, seed=42):
    """
    生成含电压恢复特征的PEMFC数据（复用RAMM逻辑，适配多步长）
    :param n_samples: 样本总数
    :param seed: 随机种子
    :return: DataFrame，包含Voltage、Current、Temperature等6维特征
    """
    np.random.seed(seed)
    # 基础特征生成
    time = np.arange(n_samples)
    current = 0.5 + 0.2 * np.sin(time / 100) + 0.1 * np.random.randn(n_samples)  # 负载波动
    temperature = 65 + 5 * np.sin(time / 500) + 1 * np.random.randn(n_samples)  # 温度扰动
    humidity = 80 + 8 * np.random.randn(n_samples)  # 湿度
    pressure = 101.3 + 2 * np.random.randn(n_samples)  # 压力
    degradation_trend = 0.0001 * time + 0.00005 * np.cumsum(np.random.randn(n_samples) / 100)  # 长期退化

    # 电压计算（含恢复行为，RAMM核心特征）
    base_voltage = 0.7 - degradation_trend
    transient_voltage = 0.05 * np.sin(current * 10) + 0.02 * np.random.randn(n_samples)  # 短期波动
    recovery_effect = np.where(time % 1000 < 200, 0.01 * (200 - time % 1000) / 200, 0)  # 恢复行为
    voltage = base_voltage + transient_voltage + recovery_effect

    # 构建DataFrame
    data = pd.DataFrame({
        "Time": time,
        "Voltage": voltage,
        "Current": current,
        "Temperature": temperature,
        "Humidity": humidity,
        "Pressure": pressure,
        "Degradation": degradation_trend
    })
    return data


def preprocess_data_eamff(raw_data, window_len=500, step=5, predict_step=10, test_size=0.3, seed=42):
    """
    预处理数据：滑动窗口划分、归一化、训练/测试集拆分（适配多步长）
    :param raw_data: 原始DataFrame
    :param window_len: 输入窗口长度I=500
    :param step: 滑动步长
    :param predict_step: 预测步长O
    :param test_size: 测试集比例
    :param seed: 随机种子
    :return: scaler, X_train, X_test, y_train, y_test
    """
    # 提取特征（移除Time列）
    features = raw_data.drop("Time", axis=1).values
    feature_names = raw_data.drop("Time", axis=1).columns.tolist()
    voltage_idx = feature_names.index("Voltage")

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # 滑动窗口生成样本
    X, y = [], []
    max_idx = len(features_scaled) - window_len - predict_step + 1
    for i in range(0, max_idx, step):
        # 输入：window_len长度的序列
        X.append(features_scaled[i:i + window_len])
        # 目标：predict_step长度的序列（重点是Voltage）
        y.append(features_scaled[i + window_len:i + window_len + predict_step])

    X = np.array(X)
    y = np.array(y)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True
    )

    return scaler, X_train, X_test, y_train, y_test


# 独立调试代码
if __name__ == "__main__":
    print("=== 调试data_process_eamff.py ===")
    # 生成数据
    data = generate_fuel_cell_data_eamff(n_samples=8000, seed=42)
    print(f"原始数据形状: {data.shape}")
    print(f"数据列名: {data.columns.tolist()}")
    print(f"电压均值: {data['Voltage'].mean():.4f}, 标准差: {data['Voltage'].std():.4f}")

    # 预处理（测试O=10步）
    scaler, X_train, X_test, y_train, y_test = preprocess_data_eamff(
        data, window_len=500, step=5, predict_step=10, seed=42
    )
    print(f"\n预处理后 - 训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"预处理后 - 测试集: X={X_test.shape}, y={y_test.shape}")
    print(f"输入序列维度: (样本数, 窗口长度, 特征数) = {X_train.shape}")
    print(f"目标序列维度: (样本数, 预测步长, 特征数) = {y_train.shape}")