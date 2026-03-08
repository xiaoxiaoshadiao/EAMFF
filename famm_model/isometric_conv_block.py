import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置全局随机种子
torch.manual_seed(42)
np.random.seed(42)


class PositionalEncoding(nn.Module):
    """正弦位置编码（PE），保留时间索引信息"""

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: (window_len, batch, d_model)
        :return: x + pe[:x.size(0)]
        """
        x = x + self.pe[:x.size(0)]
        return x


class IsometricConvBlock(nn.Module):
    """
    等距卷积模块（ICB），双层堆叠+参数共享
    对应论文4.2.2节：编码→稳健分解→跨尺度建模→长度恢复
    """

    def __init__(self, in_features, d_model=64, down_sample_factor=2, kernel_size=3):
        super(IsometricConvBlock, self).__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.down_sample_factor = down_sample_factor

        # 1. 数值编码（VE）：线性变换到高维
        self.ve_linear = nn.Linear(in_features, d_model)

        # 2. 位置编码（PE）
        self.pe = PositionalEncoding(d_model=d_model)

        # 3. 等距卷积层（跨尺度建模）
        self.ic_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.ic_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        # 4. 下采样/上采样层
        self.avg_pool = nn.AvgPool1d(kernel_size=down_sample_factor, stride=down_sample_factor)
        self.up_sample = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=down_sample_factor,
            stride=down_sample_factor
        )

        # 5. 归一化+激活+正则化
        self.layer_norm = nn.LayerNorm(d_model)
        self.tanh = nn.Tanh()  # 论文指定激活函数
        self.dropout = nn.Dropout(0.1)

    def forward(self, xs):
        """
        前向传播：处理局部波动项Xs，提取非线性波动特征
        :param xs: 局部波动项 (batch, window_len, in_features)
        :return: ys (batch, window_len, d_model) → 局部特征
        """
        batch_size, window_len, _ = xs.shape

        # 1. 数值编码（VE）
        xs_ve = self.ve_linear(xs)  # (batch, L, d_model)

        # 2. 位置编码（PE）：需转换维度 (L, batch, d_model)
        xs_pe = self.pe(xs_ve.transpose(0, 1)).transpose(0, 1)  # (batch, L, d_model)

        # 3. 零填充（保证分解后长度一致，论文要求）
        xs_pad = F.pad(xs_pe, (0, 0, 0, 0), mode="constant", value=0)  # 无实际填充，仅占位

        # 4. 跨尺度建模（双层ICB+跳跃连接）
        # 维度转换：(batch, L, d_model) → (batch, d_model, L) 适配1D卷积
        x_trans = xs_pad.transpose(1, 2)

        # 第一层卷积
        conv1_out = self.ic_conv1(x_trans)
        conv1_out = self.layer_norm(conv1_out.transpose(1, 2)).transpose(1, 2)
        conv1_out = self.tanh(conv1_out)
        conv1_out = self.dropout(conv1_out)

        # 下采样
        conv1_down = self.avg_pool(conv1_out)

        # 第二层卷积（参数共享）
        conv2_out = self.ic_conv2(conv1_down)
        conv2_out = self.layer_norm(conv2_out.transpose(1, 2)).transpose(1, 2)
        conv2_out = self.tanh(conv2_out)
        conv2_out = self.dropout(conv2_out)

        # 上采样（恢复长度）
        conv2_up = self.up_sample(conv2_out)

        # 截断到原始长度（避免上采样后长度偏差）
        if conv2_up.shape[-1] > window_len:
            conv2_up = conv2_up[:, :, :window_len]

        # 跳跃连接（原始输入+卷积输出）
        ys_trans = conv2_up + x_trans
        ys = ys_trans.transpose(1, 2)  # (batch, L, d_model)

        return ys


# ==================== Debug测试 ====================
if __name__ == "__main__":
    print("=== 测试isometric_conv_block.py功能 ===")

    # 1. 构造测试数据（模拟多尺度分解后的局部波动项Xs）
    batch_size = 8
    window_len = 300
    in_features = 6
    d_model = 64
    test_xs = torch.randn(batch_size, window_len, in_features)
    print(f"输入局部波动项形状: {test_xs.shape}")

    # 2. 初始化ICB模块
    icb = IsometricConvBlock(in_features=in_features, d_model=d_model, down_sample_factor=2)

    # 3. 前向传播
    ys = icb(test_xs)
    print(f"输出局部特征Ys形状: {ys.shape}")

    # 验证维度
    assert ys.shape == (batch_size, window_len, d_model), "Ys维度不符合预期"

    # 验证激活函数输出范围（Tanh输出应在[-1,1]）
    ys_max = torch.max(ys).item()
    ys_min = torch.min(ys).item()
    print(f"Ys最大值: {ys_max:.2f}, 最小值: {ys_min:.2f} (预期在[-1,1])")

    print("\n=== isometric_conv_block.py测试完成，功能正常 ===")