import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def generate_fuel_cell_data_ramm(n_samples=5000, seed=42):
    """
    生成含电压恢复特征的燃料电池虚拟数据（对齐论文RAMM需求）
    :param n_samples: 总采样点数
    :param seed: 随机种子
    :return: 包含6维特征的DataFrame（Voltage, Current, Temp_anode, Resistance, Humidity, Pressure）
    """
    np.random.seed(seed)

    # 基础时间序列
    t = np.linspace(0, 100, n_samples)

    # 1. 长期退化趋势（指数衰减）
    base_voltage = 0.85 * np.exp(-t / 500)  # 基础电压退化
    current = 0.2 + 0.1 * np.sin(t / 10)  # 电流（负载波动）
    temp_anode = 60 + 5 * np.sin(t / 20)  # 阳极温度
    resistance = 0.1 + 0.05 * np.exp(t / 500)  # 内阻（随退化增大）
    humidity = 80 + 10 * np.cos(t / 15)  # 湿度
    pressure = 1.0 + 0.1 * np.sin(t / 25)  # 压力

    # 2. 电压恢复特征（每100个采样点触发1次，恢复幅度随退化递减）
    recovery_amplitude = 0.05 * np.exp(-t / 500)  # 恢复幅度递减
    recovery_mask = np.zeros(n_samples)
    # 每100个采样点触发恢复（持续10个采样点）
    for i in range(100, n_samples, 100):
        if i + 10 < n_samples:
            recovery_mask[i:i + 10] = 1
    voltage_recovery = recovery_mask * recovery_amplitude * np.random.randn(n_samples) * 0.5
    # 恢复行为：先降后升
    voltage_recovery = np.convolve(voltage_recovery, np.array([-0.5, -0.3, 0, 0.3, 0.5]), mode='same')

    # 最终电压 = 基础退化 + 恢复 + 噪声
    voltage = base_voltage + voltage_recovery + 0.005 * np.random.randn(n_samples)

    # 构建DataFrame
    data = pd.DataFrame({
        'Voltage': voltage,
        'Current': current,
        'Temp_anode': temp_anode,
        'Resistance': resistance,
        'Humidity': humidity,
        'Pressure': pressure
    })

    return data


def sliding_window_split(data, window_len=500, step=5, predict_step=100):
    """
    滑动窗口划分长序列，构建训练样本（输入序列→长步长预测目标）
    :param data: 归一化后的数组 (n_samples, n_features)
    :param window_len: 输入窗口长度
    :param step: 滑动步长
    :param predict_step: 预测步长（未来步数）
    :return: X (n_samples, window_len, n_features), y (n_samples, predict_step, n_features)
    """
    X, y = [], []
    max_idx = len(data) - window_len - predict_step + 1
    for i in range(0, max_idx, step):
        # 输入窗口：i ~ i+window_len
        X.append(data[i:i + window_len])
        # 预测目标：i+window_len ~ i+window_len+predict_step
        y.append(data[i + window_len:i + window_len + predict_step])
    return np.array(X), np.array(y)


def preprocess_data_ramm(raw_data, window_len=500, step=5, predict_step=100, test_size=0.3, seed=42):
    """
    数据预处理：归一化、趋势平稳化、滑动窗口、划分训练/测试集
    :param raw_data: 原始DataFrame
    :param window_len: 输入窗口长度
    :param step: 滑动步长
    :param predict_step: 预测步长
    :param test_size: 测试集比例
    :param seed: 随机种子
    :return: scaler, X_train, X_test, y_train, y_test
    """
    # 1. 趋势平稳化（一阶差分）
    data_diff = raw_data.diff().dropna()

    # 2. Min-Max归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_diff)

    # 3. 滑动窗口划分
    X, y = sliding_window_split(data_scaled, window_len, step, predict_step)

    # 4. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=False  # 时序数据不打乱
    )

    return scaler, X_train, X_test, y_train, y_test


# 独立调试代码
if __name__ == "__main__":
    # 测试数据生成
    print("测试数据生成...")
    raw_data = generate_fuel_cell_data_ramm(n_samples=5000)
    print(f"原始数据形状: {raw_data.shape}")
    print("原始数据前5行：")
    print(raw_data.head())

    # 测试预处理
    print("\n测试数据预处理...")
    scaler, X_train, X_test, y_train, y_test = preprocess_data_ramm(
        raw_data, window_len=500, step=5, predict_step=100
    )
    print(f"训练集X形状: {X_train.shape}, 训练集y形状: {y_train.shape}")
    print(f"测试集X形状: {X_test.shape}, 测试集y形状: {y_test.shape}")
    print("预处理测试完成！")