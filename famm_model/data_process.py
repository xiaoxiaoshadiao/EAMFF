import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 设置全局随机种子（保证可复现性）
np.random.seed(42)


def generate_fuel_cell_data(n_samples=1000, seed=42):
    """
    生成带短期波动的燃料电池虚拟数据（复用基础逻辑+新增高频扰动）
    :param n_samples: 采样点数量
    :param seed: 随机种子
    :return: 包含特征+电压的DataFrame
    """
    np.random.seed(seed)

    # 基础特征（14维，延续PM得分.py逻辑）
    data = pd.DataFrame()
    data["Time"] = np.linspace(0, 1000, n_samples)  # 时间（s）
    data["Current"] = np.random.uniform(0, 35.6, n_samples)  # 电流（A）
    data["Pressure_anode_inlet"] = np.random.uniform(100, 200, n_samples)  # 阳极入口压力（kPa）
    data["Pressure_cathode_inlet"] = np.random.uniform(100, 200, n_samples)  # 阴极入口压力（kPa）
    data["Temp_anode"] = np.random.uniform(60, 80, n_samples)  # 阳极温度（℃）
    data["Temp_cathode"] = np.random.uniform(60, 80, n_samples)  # 阴极温度（℃）
    data["Humidity_anode"] = np.random.uniform(0.2, 0.8, n_samples)  # 阳极湿度
    data["Humidity_cathode"] = np.random.uniform(0.2, 0.8, n_samples)  # 阴极湿度
    data["Flow_anode"] = np.random.uniform(10, 50, n_samples)  # 阳极流量（NLPM）
    data["Flow_cathode"] = np.random.uniform(50, 200, n_samples)  # 阴极流量（NLPM）
    data["Temp_dew_anode"] = np.random.uniform(50, 70, n_samples)  # 阳极露点温度（℃）
    data["Temp_dew_cathode"] = np.random.uniform(50, 70, n_samples)  # 阴极露点温度（℃）
    data["Power"] = data["Current"] * (0.6 + np.random.normal(0, 0.05, n_samples))  # 功率（W）
    data["Resistance"] = np.random.uniform(0.01, 0.05, n_samples)  # 欧姆电阻（Ω）

    # 生成电压序列（含慢变趋势+短期波动+噪声，贴合FAMM模型需求）
    # 1. 慢变退化趋势
    trend = 0.7 - 0.0001 * data["Time"].values
    # 2. 短期高频波动（模拟负载突变/水热扰动）
    short_fluct = 0.02 * np.sin(0.1 * data["Time"].values) + 0.015 * np.cos(0.05 * data["Time"].values)
    # 3. 传感器噪声
    noise = np.random.normal(0, 0.008, n_samples)
    # 合成电压
    data["Voltage"] = trend + short_fluct + noise

    # 筛选P-M关键特征（6-8维，贴合论文）
    key_features = ["Current", "Temp_anode", "Resistance", "Pressure_cathode_inlet", "Power", "Voltage"]
    data = data[key_features]

    return data


def sliding_window_split(data, window_len=300, step=10, predict_step=1):
    """
    滑动窗口划分序列数据，构建训练样本
    :param data: 归一化后的特征数据（np.array，shape=(n_samples, n_features)）
    :param window_len: 输入序列长度I（默认300）
    :param step: 窗口滑动步长（默认10）
    :param predict_step: 预测步长O（默认1）
    :return: X (n_samples, window_len, n_features), y (n_samples, predict_step, n_features)
    """
    X, y = [], []
    n_samples, n_features = data.shape

    # 遍历所有有效窗口
    for i in range(0, n_samples - window_len - predict_step + 1, step):
        # 输入窗口：[i, i+window_len)
        X.append(data[i:i + window_len])
        # 预测目标：[i+window_len, i+window_len+predict_step)
        y.append(data[i + window_len:i + window_len + predict_step])

    return np.array(X), np.array(y)


# def preprocess_data(data, test_size=0.3, window_len=300, step=10, predict_step=1):
#     """
#     完整数据预处理流程：归一化→滑动窗口→划分训练/测试集
#     :param data: 原始DataFrame
#     :param test_size: 测试集比例（默认0.3）
#     :param window_len: 输入序列长度
#     :param step: 滑动步长
#     :param predict_step: 预测步长
#     :return: 归一化器+训练/测试集（X_train, X_test, y_train, y_test）
#     """
#     # 1. 分离特征和目标（此处所有特征都参与预测，核心关注Voltage）
#     feature_cols = data.columns.tolist()
#     data_np = data.values
#
#     # 2. Min-Max归一化（按特征独立归一化）
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data_np)
#
#     # 3. 滑动窗口划分序列
#     X, y = sliding_window_split(data_scaled, window_len=window_len, step=step, predict_step=predict_step)
#
#     # 4. 划分训练/测试集（7:3）
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
#
#     return scaler, X_train, X_test, y_train, y_test


def preprocess_data(raw_data, window_len=300, step=10, predict_step=1):
    """
    数据预处理：归一化 + 滑动窗口划分训练/测试集
    """
    # 归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)  # 拟合并转换
    scaler.feature_names_in_ = raw_data.columns.values  # 新增：手动设置特征名属性

    # 滑动窗口构建样本
    X, y = [], []
    for i in range(window_len, len(scaled_data) - predict_step + 1, step):
        X.append(scaled_data[i - window_len:i])
        y.append(scaled_data[i + predict_step - 1:i + predict_step])

    X = np.array(X)
    y = np.array(y)

    # 划分训练测试集（8:2）
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return scaler, X_train, X_test, y_train, y_test


# ==================== Debug测试 ====================
if __name__ == "__main__":
    print("=== 测试data_process.py功能 ===")

    # 1. 测试数据生成
    print("\n1. 生成燃料电池数据...")
    raw_data = generate_fuel_cell_data(n_samples=2000)
    print(f"原始数据形状: {raw_data.shape}")
    print(f"原始数据前5行:\n{raw_data.head()}")
    print(f"电压范围: {raw_data['Voltage'].min():.3f} ~ {raw_data['Voltage'].max():.3f}")

    # 2. 测试数据预处理
    print("\n2. 数据预处理（归一化+滑动窗口）...")
    scaler, X_train, X_test, y_train, y_test = preprocess_data(
        raw_data, window_len=300, step=10, predict_step=1
    )
    print(f"训练集输入形状 (batch, window_len, features): {X_train.shape}")
    print(f"训练集输出形状 (batch, predict_step, features): {y_train.shape}")
    print(f"测试集输入形状: {X_test.shape}")
    print(f"测试集输出形状: {y_test.shape}")

    # 验证归一化效果
    print("\n3. 验证归一化效果...")
    train_voltage = X_train[:, :, raw_data.columns.get_loc("Voltage")]
    print(f"归一化后电压范围: {train_voltage.min():.3f} ~ {train_voltage.max():.3f}")

    print("\n=== data_process.py测试完成，功能正常 ===")