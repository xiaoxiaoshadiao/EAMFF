# 基础库
import numpy as np
import pandas as pd
# 计算相关库
from scipy.stats import pearsonr  # 用于P-Score（Pearson相关系数）
from sklearn.feature_selection import mutual_info_regression  # 用于M-Score（互信息）
# 可视化库（可选，方便查看结果）
import matplotlib.pyplot as plt

# 解决中文显示问题（可选，避免图表乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def generate_fuel_cell_data(n_samples=1000, seed=42):
    """
    生成模拟燃料电池运行数据，包含1个目标变量（Voltage）和14个特征变量
    特征变量参考论文数据集（FC-DLC/IEEE PHM/215通道）：电流、压力、温度、湿度、流量等
    """
    print("正在生成模拟燃料电池数据...")  # 新增日志
    np.random.seed(seed)

    # 生成基础特征（14维）
    data = pd.DataFrame()
    data["Time"] = np.linspace(0, 1000, n_samples)  # 时间（s）
    data["Current"] = np.random.uniform(0, 35.6, n_samples)  # 电流（A，参考FC-DLC工况）
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

    # 生成目标变量：Voltage（受特征变量影响，模拟退化趋势）
    data["Voltage"] = (
            0.7  # 基准电压
            - 0.02 * data["Current"]  # 电流负相关
            - 0.001 * data["Resistance"]  # 电阻负相关
            + 0.005 * data["Temp_anode"]  # 温度正相关
            - 0.003 * (data["Pressure_cathode_inlet"] - 150)  # 阴极压力偏离基准的影响
            + np.random.normal(0, 0.01, n_samples)  # 随机噪声
    )
    print("数据生成完成！共生成 {} 条样本，14 个特征变量".format(n_samples))  # 新增日志
    return data


# 调用生成数据
df = generate_fuel_cell_data()
X = df.drop("Voltage", axis=1)  # 14维特征
y = df["Voltage"]  # 目标变量（电压，退化表征指标）


def calculate_p_score(X, y):
    """
    计算每个特征与目标变量y的Pearson相关系数（P-Score）
    论文公式3-2，取值范围[-1,1]，绝对值越大线性相关性越强
    """
    print("\n正在计算P-Score（Pearson相关系数）...")  # 新增日志
    p_scores = {}
    for col in X.columns:
        # 计算Pearson相关系数（返回系数和p值，取系数绝对值作为P-Score）
        corr, _ = pearsonr(X[col], y)
        p_scores[col] = abs(corr)  # 取绝对值，只关注相关性强度
    return pd.Series(p_scores, name="P_Score")


def calculate_m_score(X, y):
    """
    计算每个特征与目标变量y的互信息（M-Score）
    论文公式3-10，取值≥0，值越大非线性相关性越强
    """
    print("正在计算M-Score（互信息）...")  # 新增日志
    # 修复：移除新版sklearn不支持的normalize参数
    m_scores = mutual_info_regression(X, y, random_state=42)
    # 手动归一化（替代原来的normalize=True，保证量纲统一）
    m_scores = (m_scores - m_scores.min()) / (m_scores.max() - m_scores.min())
    return pd.Series(m_scores, index=X.columns, name="M_Score")


def calculate_pm_score(p_scores, m_scores, A=0.5, B=0.5):
    """
    计算PM综合得分：Score = A*M_Score + B*P_Score（论文公式3-11）
    A、B为权重（可调整，默认各0.5，兼顾线性和非线性）
    """
    print("正在计算PM综合得分...")  # 新增日志
    # 确保两个Series索引一致
    pm_scores = A * m_scores + B * p_scores
    return pm_scores.sort_values(ascending=False)


def select_features_by_pm(pm_scores, top_k=None, threshold=None):
    """
    根据PM得分筛选特征：
    - top_k：筛选前k个得分最高的特征（二选一）
    - threshold：筛选得分≥threshold的特征
    """
    if top_k is not None:
        selected_features = pm_scores.head(top_k).index.tolist()
    elif threshold is not None:
        selected_features = pm_scores[pm_scores >= threshold].index.tolist()
    else:
        raise ValueError("必须指定top_k或threshold")
    return selected_features


if __name__ == "__main__":
    # 1. 生成数据（已执行）
    # 2. 计算P-Score
    p_scores = calculate_p_score(X, y)
    # 3. 计算M-Score
    m_scores = calculate_m_score(X, y)
    # 4. 计算PM综合得分
    pm_scores = calculate_pm_score(p_scores, m_scores, A=0.5, B=0.5)
    # 5. 筛选特征（示例：选前8个得分最高的特征）
    selected_features = select_features_by_pm(pm_scores, top_k=8)

    # 输出结果
    print("\n=== P-M得分排名（降序）===")
    result_df = pd.DataFrame({
        "P_Score": p_scores,
        "M_Score": m_scores,
        "PM_Score": pm_scores
    }).sort_values("PM_Score", ascending=False)
    print(result_df)

    print("\n=== 筛选后的特征（Top8）===")
    print(selected_features)

    # （可选）可视化Top10特征的PM得分
    print("\n正在生成PM得分可视化图表...")
    plt.figure(figsize=(12, 6))
    pm_scores.head(10).plot(kind="bar", color="#1f77b4")
    plt.title("P-M Score Top10 特征", fontsize=14)
    plt.xlabel("特征名称", fontsize=12)
    plt.ylabel("PM 综合得分", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("PM_Score_Top10.png")  # 保存图表到本地
    plt.show()
    print("\n所有计算完成！图表已保存为 PM_Score_Top10.png")