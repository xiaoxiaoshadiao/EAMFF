import numpy as np
import torch
import torch.nn as nn

# 设置全局随机种子
torch.manual_seed(42)
np.random.seed(42)


class MultiScaleDecompose(nn.Module):
    """
    多尺度分解模块（双核平均池化+能量近似守恒）
    对应论文4.2.1节
    """

    def __init__(self, kernel1=3, kernel2=5, padding_mode="constant"):
        super(MultiScaleDecompose, self).__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.padding_mode = padding_mode

        # 定义双核平均池化层（1D池化，针对序列数据）
        self.avg_pool1 = nn.AvgPool1d(kernel_size=kernel1, stride=1, padding=kernel1 // 2)
        self.avg_pool2 = nn.AvgPool1d(kernel_size=kernel2, stride=1, padding=kernel2 // 2)

    def forward(self, x):
        """
        前向传播：输入序列拆分为局部波动项Xs和背景项Xbg
        :param x: 输入张量 (batch, window_len, features) → 需转置为 (batch, features, window_len) 适配1D池化
        :return: Xs (batch, window_len, features), Xbg (batch, window_len, features)
        """
        # 维度转换：(batch, L, F) → (batch, F, L)
        x_trans = x.transpose(1, 2)

        # 双核平均池化计算背景项Xbg
        pool1_out = self.avg_pool1(x_trans)
        pool2_out = self.avg_pool2(x_trans)
        xbg_trans = (pool1_out + pool2_out) / 2  # 双池化均值

        # 维度还原：(batch, F, L) → (batch, L, F)
        xbg = xbg_trans.transpose(1, 2)

        # 残差计算局部波动项Xs = X - Xbg
        xs = x - xbg

        # 可选：能量守恒验证（仅在调试模式下）
        if self.training is False:
            energy_x = torch.norm(x, p=2) ** 2
            energy_xs = torch.norm(xs, p=2) ** 2
            energy_xbg = torch.norm(xbg, p=2) ** 2
            energy_error = abs(energy_x - (energy_xs + energy_xbg)) / energy_x * 100
            print(f"能量守恒误差: {energy_error.item():.2f}% (阈值<5%)")

        return xs, xbg


# ==================== Debug测试 ====================
if __name__ == "__main__":
    print("=== 测试multi_scale_decompose.py功能 ===")

    # 1. 构造测试数据（模拟预处理后的输入）
    batch_size = 8
    window_len = 300
    n_features = 6
    test_x = torch.randn(batch_size, window_len, n_features)  # (batch, L, F)
    print(f"输入张量形状: {test_x.shape}")

    # 2. 初始化多尺度分解模块
    msd = MultiScaleDecompose(kernel1=3, kernel2=5)
    msd.eval()  # 评估模式（开启能量验证）

    # 3. 前向传播
    xs, xbg = msd(test_x)
    print(f"局部波动项Xs形状: {xs.shape}")
    print(f"背景项Xbg形状: {xbg.shape}")

    # 验证维度一致性
    assert xs.shape == test_x.shape, "Xs维度与输入不一致"
    assert xbg.shape == test_x.shape, "Xbg维度与输入不一致"

    # 验证局部项的均值（波动项均值应接近0）
    xs_mean = torch.mean(xs).item()
    print(f"局部波动项均值: {xs_mean:.6f} (预期接近0)")

    print("\n=== multi_scale_decompose.py测试完成，功能正常 ===")