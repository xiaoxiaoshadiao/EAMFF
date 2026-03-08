# 基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt  # 小波变换核心库
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_noisy_fuel_cell_voltage(n_samples=1000, seed=42):
    """
    生成带噪声的燃料电池电压序列（模拟论文中传感器噪声+环境扰动）
    贴合论文FC-DLC数据集特性：电压基线0.5-0.8V，含长期退化趋势+短期波动+随机噪声
    """
    np.random.seed(seed)
    # 时间序列
    time = np.linspace(0, 1000, n_samples)

    # 1. 长期退化趋势（模拟膜衰减、催化剂老化）
    trend = 0.7 - 0.0001 * time  # 缓慢下降趋势

    # 2. 短期波动（模拟负载变化、水热扰动）
    fluctuation = 0.02 * np.sin(0.05 * time) + 0.015 * np.cos(0.02 * time)

    # 3. 随机噪声（模拟传感器误差、环境干扰，噪声强度贴合论文真实数据）
    noise = np.random.normal(0, 0.008, n_samples)

    # 合成带噪声的电压序列（论文核心退化表征指标）
    voltage_noisy = trend + fluctuation + noise
    # 生成纯净电压序列（用于计算去噪误差）
    voltage_clean = trend + fluctuation

    return pd.DataFrame({
        'Time': time,
        'Voltage_Noisy': voltage_noisy,
        'Voltage_Clean': voltage_clean
    })


def dwt_denoising(voltage_series, wavelet='db3', level=4):
    """
    论文3.2.2节指定的小波去噪流程：DWT分解+软阈值处理+重构
    :param voltage_series: 带噪声的电压序列
    :param wavelet: 小波基（论文用db3）
    :param level: 分解层数（论文用4层）
    :return: 去噪后的电压序列
    """
    # 1. 小波分解（论文公式3-12）
    # coeffs = (cA4, cD4, cD3, cD2, cD1)：cA为逼近系数（低频趋势），cD为细节系数（高频噪声）
    coeffs = pywt.wavedec(voltage_series, wavelet=wavelet, level=level)
    cA, cD4, cD3, cD2, cD1 = coeffs  # 4层分解，得到1个逼近系数+4个细节系数

    # 2. 软阈值处理细节系数（论文公式3-14）
    def soft_threshold(coeff, threshold):
        """软阈值函数：|coeff| > threshold 则保留并收缩，否则置0"""
        return np.where(np.abs(coeff) > threshold, np.sign(coeff) * (np.abs(coeff) - threshold), 0)

    # 计算自适应阈值（基于噪声强度，贴合论文逻辑）
    sigma = np.median(np.abs(cD1)) / 0.6745  # 估计噪声标准差
    threshold = sigma * np.sqrt(2 * np.log(len(voltage_series)))  # 自适应阈值

    # 对所有细节系数应用软阈值（抑制噪声）
    cD4_denoised = soft_threshold(cD4, threshold)
    cD3_denoised = soft_threshold(cD3, threshold)
    cD2_denoised = soft_threshold(cD2, threshold)
    cD1_denoised = soft_threshold(cD1, threshold)

    # 3. 重构信号（论文公式3-12逆过程）
    coeffs_denoised = (cA, cD4_denoised, cD3_denoised, cD2_denoised, cD1_denoised)
    voltage_denoised = pywt.waverec(coeffs_denoised, wavelet=wavelet)

    # 确保重构后长度与原始序列一致（避免边界效应导致的长度偏差）
    if len(voltage_denoised) != len(voltage_series):
        voltage_denoised = voltage_denoised[:len(voltage_series)]

    return voltage_denoised, threshold


def calculate_denoising_metrics(clean, noisy, denoised):
    """计算去噪效果评价指标（贴合论文误差分析逻辑）"""
    metrics = {
        'Noisy_MAE': mean_absolute_error(clean, noisy),
        'Noisy_RMSE': np.sqrt(mean_squared_error(clean, noisy)),
        'Denoised_MAE': mean_absolute_error(clean, denoised),
        'Denoised_RMSE': np.sqrt(mean_squared_error(clean, denoised)),
        'RMSE_Reduction_Rate': (np.sqrt(mean_squared_error(clean, noisy)) - np.sqrt(
            mean_squared_error(clean, denoised))) / np.sqrt(mean_squared_error(clean, noisy)) * 100
    }
    return metrics


if __name__ == "__main__":
    # 1. 生成带噪声的燃料电池电压数据
    print("=== 生成带噪声的燃料电池电压数据 ===")
    df = generate_noisy_fuel_cell_voltage(n_samples=1000, seed=42)
    time = df['Time'].values
    voltage_clean = df['Voltage_Clean'].values
    voltage_noisy = df['Voltage_Noisy'].values
    print(f"数据生成完成：{len(time)}个采样点，电压范围：{voltage_noisy.min():.3f}V ~ {voltage_noisy.max():.3f}V")

    # 2. 执行小波去噪（论文指定参数）
    print("\n=== 执行小波去噪（db3小波基+4层分解）===")
    voltage_denoised, threshold = dwt_denoising(voltage_noisy, wavelet='db3', level=4)
    print(f"自适应阈值：{threshold:.6f}")

    # 3. 计算去噪效果指标
    metrics = calculate_denoising_metrics(voltage_clean, voltage_noisy, voltage_denoised)
    print("\n=== 去噪效果评价指标 ===")
    for key, value in metrics.items():
        if 'Rate' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.6f}")

    # 4. 可视化去噪前后对比（贴合论文图3.4-3.7格式）
    print("\n=== 生成去噪前后对比图 ===")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 子图1：原始纯净信号 + 带噪声信号
    ax1.plot(time, voltage_clean, color='#2E86AB', linewidth=2, label='纯净电压信号（无噪声）')
    ax1.plot(time, voltage_noisy, color='#F24236', linewidth=1.5, alpha=0.7, label='带噪声电压信号')
    ax1.set_ylabel('电压 (V)', fontsize=12)
    ax1.set_title('小波去噪前：纯净信号 vs 带噪声信号', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # 子图2：带噪声信号 + 去噪后信号
    ax2.plot(time, voltage_noisy, color='#F24236', linewidth=1.5, alpha=0.7, label='带噪声电压信号')
    ax2.plot(time, voltage_denoised, color='#3E92CC', linewidth=2, label='DWT去噪后信号（db3+4层）')
    ax2.set_xlabel('时间 (s)', fontsize=12)
    ax2.set_ylabel('电压 (V)', fontsize=12)
    ax2.set_title(f'小波去噪后：带噪声信号 vs 去噪信号（RMSE降低{metrics["RMSE_Reduction_Rate"]:.2f}%）', fontsize=14,
                  fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('wavelet_denoising_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("对比图已保存为：wavelet_denoising_comparison.png")

    # 5. 保存结果到CSV（方便后续分析）
    result_df = pd.DataFrame({
        'Time': time,
        'Voltage_Clean': voltage_clean,
        'Voltage_Noisy': voltage_noisy,
        'Voltage_Denoised': voltage_denoised
    })
    result_df.to_csv('wavelet_denoising_result.csv', index=False, encoding='utf-8-sig')
    print("去噪结果已保存为：wavelet_denoising_result.csv")