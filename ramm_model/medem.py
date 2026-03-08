import torch
import torch.nn as nn
import numpy as np


class ExponentialDecomposition(nn.Module):
    """指数下采样分解（论文公式5-2）"""

    def __init__(self, m=3):
        super().__init__()
        self.m = m  # 分解层数，生成m+1个子序列

    def forward(self, x):
        """
        :param x: 输入张量 (batch, window_len, in_features)
        :return: 多尺度子序列列表 [x0, x1, ..., xm]，长度依次减半
        """
        batch_size, window_len, in_features = x.shape
        sub_sequences = [x]  # x0 = 原始序列

        # 指数下采样（平均池化，长度依次减半）
        for i in range(1, self.m + 1):
            pool_kernel = 2 ** i
            # 平均池化保证趋势平滑
            pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel, padding=0)
            # 调整维度：(batch, in_features, window_len) → 池化 → (batch, in_features, new_len) → 转置
            x_pooled = pool(x.transpose(1, 2)).transpose(1, 2)
            sub_sequences.append(x_pooled)

        return sub_sequences


class Encoding(nn.Module):
    """双重编码（数值编码VE + 位置编码PE，论文公式5-3）"""

    def __init__(self, in_features, f_enc=128):
        super().__init__()
        self.f_enc = f_enc
        # 数值编码：全连接层线性映射
        self.ve = nn.Linear(in_features, f_enc)
        # 层归一化保证数值稳定性
        self.norm = nn.LayerNorm(f_enc)

    def positional_encoding(self, seq_len, device):
        """正弦位置编码（修复设备不一致问题）"""
        pe = torch.zeros(seq_len, self.f_enc, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        # 关键：将div_term移到指定device上
        div_term = torch.exp(
            torch.arange(0, self.f_enc, 2).float() * (-np.log(10000.0) / self.f_enc)
        ).to(device)  # 新增to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, f_enc)

    def forward(self, sub_sequences):
        """
        :param sub_sequences: 指数分解后的子序列列表
        :return: 编码后的子序列列表 [x0_enc, x1_enc, ..., xm_enc]
        """
        encoded_subs = []
        for sub in sub_sequences:
            batch_size, seq_len, _ = sub.shape
            # 1. 数值编码
            ve_out = self.ve(sub)
            # 2. 位置编码
            pe = self.positional_encoding(seq_len, sub.device)
            # 3. 编码融合 + 归一化
            enc = self.norm(ve_out + pe)
            encoded_subs.append(enc)
        return encoded_subs


class MEDEM(nn.Module):
    """多尺度指数分解编码模块（MEDEM）"""

    def __init__(self, in_features, m=3, f_enc=128):
        super().__init__()
        self.exp_decomp = ExponentialDecomposition(m=m)
        self.encoding = Encoding(in_features=in_features, f_enc=f_enc)

    def forward(self, x):
        """
        :param x: 输入张量 (batch, window_len, in_features)
        :return: 编码后的多尺度子序列列表 X_E^0
        """
        # 1. 指数下采样分解
        sub_sequences = self.exp_decomp(x)
        # 2. 双重编码
        encoded_subs = self.encoding(sub_sequences)
        return encoded_subs


# 独立调试代码
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 测试参数
    batch_size = 16
    window_len = 500
    in_features = 6
    m = 3
    f_enc = 128

    # 生成测试数据
    x = torch.randn(batch_size, window_len, in_features).to(device)
    print(f"输入形状: {x.shape}")

    # 初始化MEDEM
    medem = MEDEM(in_features=in_features, m=m, f_enc=f_enc).to(device)

    # 前向传播
    encoded_subs = medem(x)

    # 输出测试结果
    print("\n指数分解+编码后的子序列形状：")
    for i, sub in enumerate(encoded_subs):
        print(f"X_{i}^0: {sub.shape}")
    print("MEDEM模块测试完成！")