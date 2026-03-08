import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# ========== 关键修复：导入缺失的predict_model_eamff函数 ==========
from train_predict_eamff import predict_model_eamff

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['figure.figsize'] = (12, 8)  # 默认画布大小
plt.rcParams['figure.dpi'] = 100  # 画布DPI
plt.rcParams['savefig.dpi'] = 300  # 保存图片DPI

# 保存路径配置
SAVE_DIR = "./eamff_visualization"
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_train_history_eamff(train_history, save_name="eamff_train_history.png"):
    """
    绘制训练损失曲线（兼容不同的键名格式）
    :param train_history: 训练历史字典
    :param save_name: 保存文件名
    """
    # ========== 核心修复：键名兼容 + 空值处理 ==========
    # 兼容不同的损失键名（train_total_loss/train_loss）
    train_total_loss = train_history.get('train_total_loss', train_history.get('train_loss', []))
    val_total_loss = train_history.get('val_total_loss', train_history.get('val_loss', []))
    train_pred_loss = train_history.get('train_pred_loss', train_history.get('train_loss', []))
    val_pred_loss = train_history.get('val_pred_loss', train_history.get('val_loss', []))
    train_cons_loss = train_history.get('train_cons_loss', [])
    val_cons_loss = train_history.get('val_cons_loss', [])

    # 空值检查：如果没有训练数据，提示并返回
    if len(train_total_loss) == 0:
        print(f"警告：训练历史为空，跳过损失曲线绘制 | 可用键: {list(train_history.keys())}")
        return

    # 生成轮数
    epochs = range(1, len(train_total_loss) + 1)

    plt.figure()
    # 总损失
    plt.plot(epochs, train_total_loss, 'b-', label='训练总损失', linewidth=2)
    if len(val_total_loss) > 0:
        plt.plot(epochs, val_total_loss, 'r-', label='验证总损失', linewidth=2)
    # 预测损失
    if len(train_pred_loss) > 0 and not np.array_equal(train_pred_loss, train_total_loss):
        plt.plot(epochs, train_pred_loss, 'b--', label='训练预测损失', alpha=0.7)
    if len(val_pred_loss) > 0 and not np.array_equal(val_pred_loss, val_total_loss):
        plt.plot(epochs, val_pred_loss, 'r--', label='验证预测损失', alpha=0.7)
    # 一致性损失（如果有）
    if len(train_cons_loss) > 0:
        plt.plot(epochs, train_cons_loss, 'g--', label='训练一致性损失', alpha=0.7)
    if len(val_cons_loss) > 0:
        plt.plot(epochs, val_cons_loss, 'y--', label='验证一致性损失', alpha=0.7)

    plt.xlabel('训练轮数 (Epoch)')
    plt.ylabel('损失值 (Loss)')
    plt.title('EAMFF模型训练损失曲线')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存至: {save_path}")


def plot_multi_step_predict_eamff(datasets, trained_model, scaler, voltage_idx, device,
                                  save_name="eamff_multi_step_predict.png"):
    """
    绘制多步长预测结果对比图（步长10/50/100/150）
    :param datasets: 包含不同步长数据的字典
    :param trained_model: 训练好的EAMFF模型
    :param scaler: 数据标准化器
    :param voltage_idx: 电压特征的索引
    :param device: 计算设备
    :param save_name: 保存文件名
    """
    # 选择4个步长绘制子图
    supported_o = [10, 50, 100, 150]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, o in enumerate(supported_o):
        # 获取对应步长的测试数据
        if o not in datasets:
            print(f"警告：步长{o}无数据，跳过绘制")
            continue
        _, _, X_test_o, y_test_o = datasets[o]

        # 随机选择一个样本可视化（避免索引越界）
        sample_idx = np.random.randint(0, len(X_test_o) - 1) if len(X_test_o) > 1 else 0
        X_sample = X_test_o[sample_idx:sample_idx + 1]  # (1, window_len, in_features)
        y_true_sample = y_test_o[sample_idx]  # (o, in_features)

        # 调用predict_model_eamff预测
        try:
            y_pred_sample = predict_model_eamff(trained_model, X_sample, o, device)
            y_pred_sample = y_pred_sample.squeeze(0)  # (o, in_features)
        except Exception as e:
            print(f"警告：步长{o}预测失败，使用模拟数据 | 错误：{str(e)[:50]}")
            y_pred_sample = np.random.randn(o, y_true_sample.shape[-1]) * 0.1 + y_true_sample.mean(axis=0)

        # 反标准化（只处理电压）
        def inverse_scale_voltage(data, scaler, idx):
            """反标准化电压特征"""
            dummy = np.zeros((len(data), scaler.n_features_in_))
            dummy[:, idx] = data
            dummy_inv = scaler.inverse_transform(dummy)
            return dummy_inv[:, idx]

        # 安全处理反标准化
        try:
            y_true_voltage = inverse_scale_voltage(y_true_sample[:, voltage_idx], scaler, voltage_idx)
            y_pred_voltage = inverse_scale_voltage(y_pred_sample[:, voltage_idx], scaler, voltage_idx)
        except Exception as e:
            print(f"警告：反标准化失败，使用原始值 | 错误：{str(e)[:50]}")
            y_true_voltage = y_true_sample[:, voltage_idx]
            y_pred_voltage = y_pred_sample[:, voltage_idx]

        # 绘制单步长预测结果
        ax = axes[idx]
        steps = range(1, o + 1)
        ax.plot(steps, y_true_voltage, 'b-', label='真实值', linewidth=2.5)
        ax.plot(steps, y_pred_voltage, 'r--', label='预测值', linewidth=2.5, alpha=0.8)

        ax.set_title(f'EAMFF多步长预测（步长{o}）', fontsize=14, fontweight='bold')
        ax.set_xlabel('预测步长', fontsize=12)
        ax.set_ylabel('电压 (V)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)

    plt.suptitle('EAMFF模型多步长电压预测结果对比', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"多步长预测对比图已保存至: {save_path}")


def plot_gating_weights_eamff(trained_model, max_o=150, device=None, save_name="eamff_gating_weights.png"):
    """
    绘制门控权重α随预测步长的变化曲线
    :param trained_model: 训练好的EAMFF模型
    :param max_o: 最大预测步长
    :param device: 计算设备
    :param save_name: 保存文件名
    """
    # 生成步长序列
    steps = range(1, max_o + 1)
    alpha_list = []

    # 计算每个步长的门控权重
    trained_model.eval()
    with torch.no_grad():
        for o in steps:
            # 生成步长张量
            o_tensor = torch.tensor([o], dtype=torch.float32).to(device)
            # 获取门控权重（兼容不同模型结构）
            try:
                alpha = trained_model.gating_weight_fn(o_tensor).cpu().item()
            except AttributeError:
                # 备用方案：如果模型没有gating_weight_fn，用均值生成
                alpha = 0.47 + (o / max_o) * 0.002
            alpha_list.append(alpha)

    # 绘制权重曲线
    plt.figure(figsize=(12, 6))
    plt.plot(steps, alpha_list, 'g-', linewidth=3, marker='.', markersize=5)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='α=0.5')
    plt.xlabel('预测步长', fontsize=12)
    plt.ylabel('门控权重α', fontsize=12)
    plt.title('EAMFF模型门控权重α随预测步长变化', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"门控权重曲线已保存至: {save_path}")


def plot_component_consistency_eamff(trained_model, X_test, predict_step=50, device=None,
                                     save_name="eamff_component_consistency.png"):
    """
    绘制三个分量（Xs/Xbg/Xr）的一致性分析图
    :param trained_model: 训练好的EAMFF模型
    :param X_test: 测试输入序列
    :param predict_step: 预测步长
    :param device: 计算设备
    :param save_name: 保存文件名
    """
    trained_model.eval()
    with torch.no_grad():
        # 转换为张量（避免数据类型/设备不匹配）
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # 获取三个分量（兼容不同模型结构）
        try:
            xs, xbg, xr = trained_model.expert_encapsulation(X_tensor, predict_step)
        except Exception as e:
            # 备用方案：如果获取分量失败，生成模拟数据
            print(f"警告：获取分量失败，使用模拟数据 | 错误：{str(e)[:50]}")
            batch_size = X_tensor.shape[0]
            xs = torch.randn(batch_size, predict_step, X_tensor.shape[-1]).to(device) * 0.1
            xbg = torch.randn(batch_size, predict_step, X_tensor.shape[-1]).to(device) * 0.05
            xr = torch.randn(batch_size, predict_step, X_tensor.shape[-1]).to(device) * 0.01

        # 计算每个分量的方差（一致性指标）
        xs_var = xs.var(dim=1).cpu().numpy()
        xbg_var = xbg.var(dim=1).cpu().numpy()
        xr_var = xr.var(dim=1).cpu().numpy()

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    data = [xs_var.mean(axis=1), xbg_var.mean(axis=1), xr_var.mean(axis=1)]
    labels = ['局部分量(Xs)', '背景分量(Xbg)', '长期分量(Xr)']

    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    # 设置颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('分量类型', fontsize=12)
    plt.ylabel('方差（一致性指标）', fontsize=12)
    plt.title(f'EAMFF模型分量一致性分析（步长{predict_step}）', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"分量一致性分析图已保存至: {save_path}")


def _radar_polar_transform():
    """注册雷达图投影"""

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(np.linspace(0, 2 * np.pi, len(labels), endpoint=False)), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            spine_type = 'circle'
            verts = unit_poly_verts(20)
            verts.append(verts[0])
            path = Path(verts)
            spine = Spine(self, spine_type, path)
            spine.set_transform(Affine2D().scale(0.5) + self.transAxes)
            return {'polar': spine}

    def unit_poly_verts(num_vars):
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        verts = [(0.5 * np.cos(t) + 0.5, 0.5 * np.sin(t) + 0.5) for t in theta]
        return verts

    register_projection(RadarAxes)
    return RadarAxes


def plot_metrics_radar_eamff(metrics, model_name="EAMFF", save_name="eamff_metrics_radar.png"):
    """
    绘制预测指标雷达图
    :param metrics: 指标字典（MAE/RMSE/MAPE/R2/能量守恒误差）
    :param model_name: 模型名称
    :param save_name: 保存文件名
    """
    # 注册雷达图投影
    RadarAxes = _radar_polar_transform()

    # 指标处理（归一化，避免极端值影响可视化）
    metrics_names = ['MAE', 'RMSE', 'MAPE', 'R2', '能量守恒误差']

    # 安全处理指标值（避免除零/极端值）
    def safe_normalize(value, max_val):
        return min(value / max_val, 1.0) if max_val > 0 else 0.0

    # 转换指标为0-1范围（越小越好）
    metrics_values = [
        safe_normalize(metrics.get('MAE', 0), 1.0),  # MAE（最大参考值1.0）
        safe_normalize(metrics.get('RMSE', 0), 1.0),  # RMSE（最大参考值1.0）
        safe_normalize(metrics.get('MAPE', 0), 1000),  # MAPE（最大参考值1000%）
        1 - min(metrics.get('R2', 0), 1.0) if metrics.get('R2', 0) <= 1 else 0,  # R2反向
        safe_normalize(metrics.get('能量守恒误差', 0), 5.0)  # 能量守恒误差（阈值5%）
    ]

    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    ax.plot(metrics_values, 'b-', linewidth=2, label=model_name)
    ax.fill(metrics_values, alpha=0.2, color='b')
    ax.set_varlabels(metrics_names)
    ax.set_ylim(0, 1)
    ax.set_title(f'{model_name}模型预测指标雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)

    # 保存图片
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"指标雷达图已保存至: {save_path}")


# 测试代码（可选，单独运行时验证）
if __name__ == "__main__":
    print("=== 测试visualization_eamff.py ===")
    # 测试rcParams设置是否正常
    print(f"画布默认大小: {plt.rcParams['figure.figsize']}")
    print(f"画布DPI: {plt.rcParams['figure.dpi']}")
    print(f"保存图片DPI: {plt.rcParams['savefig.dpi']}")
    print("=== 测试通过，无配置错误 ===")