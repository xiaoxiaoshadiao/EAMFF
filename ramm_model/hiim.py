import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustDecomposition(nn.Module):
    """稳健分解（提取宏观趋势项+微观恢复项，论文公式5-4~5-7）"""

    def __init__(self, kernel_n=[2, 4, 6]):
        super().__init__()
        self.kernel_n = kernel_n

    def forward(self, encoded_subs):
        """
        :param encoded_subs: 编码后的子序列列表 X_E^0
        :return: 宏观项列表 X_ET^0, 微观项列表 X_EL^0
        """
        X_ET, X_EL = [], []
        for sub in encoded_subs:
            batch_size, seq_len, f_enc = sub.shape
            # 零填充保证池化后长度不小于原始长度
            pad = max(self.kernel_n) // 2
            sub_padded = F.pad(sub.transpose(1, 2), (pad, pad), mode='constant', value=0).transpose(1, 2)

            # 多核平均池化（统一输出长度为原始seq_len）
            pool_outs = []
            for kernel in self.kernel_n:
                pool = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)
                # 池化
                pool_out = pool(sub_padded.transpose(1, 2)).transpose(1, 2)
                # 统一长度到原始seq_len
                if pool_out.shape[1] > seq_len:
                    pool_out = pool_out[:, :seq_len, :]
                elif pool_out.shape[1] < seq_len:
                    pool_out = F.interpolate(
                        pool_out.transpose(1, 2),
                        size=seq_len,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                pool_outs.append(pool_out)

            # 宏观趋势项：多核池化结果取平均
            macro = torch.mean(torch.stack(pool_outs, dim=0), dim=0)
            # 微观恢复项：原始 - 宏观
            micro = sub - macro

            X_ET.append(macro)
            X_EL.append(micro)
        return X_ET, X_EL


class TrendLinear(nn.Module):
    """趋势项多尺度线性层（自上而下细化，论文公式5-8）"""

    def __init__(self, f_enc=128, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(f_enc, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, f_enc)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(f_enc)

    def forward(self, x_coarse, target_len):
        """
        :param x_coarse: 粗尺度趋势项 (batch, seq_len_coarse, f_enc)
        :param target_len: 目标细尺度长度
        :return: 上采样后的细尺度趋势项 (batch, target_len, f_enc)
        """
        batch_size, seq_len_coarse, f_enc = x_coarse.shape
        # 全连接层映射
        x = self.relu(self.fc1(x_coarse))
        x = self.fc2(x)
        # 上采样到目标长度（而非固定翻倍）
        x_upsampled = F.interpolate(
            x.transpose(1, 2),
            size=target_len,  # 匹配目标细尺度长度
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        return self.norm(x_upsampled)


class LocalLinear(nn.Module):
    """局部项多尺度线性层（自下而上聚合，论文公式5-9）"""

    def __init__(self, f_enc=128, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(f_enc, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, f_enc)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(f_enc)

    def forward(self, x_fine, target_len):
        """
        :param x_fine: 细尺度局部项 (batch, seq_len_fine, f_enc)
        :param target_len: 目标粗尺度长度
        :return: 下采样后的粗尺度局部项 (batch, target_len, f_enc)
        """
        batch_size, seq_len_fine, f_enc = x_fine.shape
        # 下采样到目标长度（而非固定减半）
        x_downsampled = F.interpolate(
            x_fine.transpose(1, 2),
            size=target_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        # 全连接层映射
        x = self.relu(self.fc1(x_downsampled))
        x = self.fc2(x)
        return self.norm(x)


class HIIMBlock(nn.Module):
    """单轮历史信息整合块（HIIM1/HIIM2，论文公式5-10~5-12）"""

    def __init__(self, m=3, f_enc=128, hidden_dim=256, kernel_n=[2, 4, 6]):
        super().__init__()
        self.m = m
        self.robust_decomp = RobustDecomposition(kernel_n=kernel_n)
        # 趋势项线性层（m个，对应m+1个子序列的自上而下更新）
        self.trend_linears = nn.ModuleList([TrendLinear(f_enc, hidden_dim) for _ in range(m)])
        # 局部项线性层（m个，对应m+1个子序列的自下而上更新）
        self.local_linears = nn.ModuleList([LocalLinear(f_enc, hidden_dim) for _ in range(m)])
        # 线性一致化层
        self.linear_fusion = nn.Linear(f_enc, f_enc)
        self.dropout = nn.Dropout(0.1)  # 防止过拟合

    def forward(self, encoded_subs):
        """
        :param encoded_subs: 编码后的子序列列表 X_E^0/X_E^1
        :return: 整合后的子序列列表 X_E^1/X_E^2
        """
        # 1. 稳健分解
        X_ET, X_EL = self.robust_decomp(encoded_subs)
        m_plus_1 = len(encoded_subs)

        # 2. 双向线性更新
        # 2.1 趋势项：自上而下（粗→细）
        for i in range(m_plus_1 - 2, -1, -1):  # 从m-1到0
            if i + 1 < m_plus_1:
                # 传入目标长度（当前细尺度的长度）
                target_len = X_ET[i].shape[1]
                trend_upsampled = self.trend_linears[i](X_ET[i + 1], target_len)
                # 最终校验：确保长度完全一致
                if trend_upsampled.shape[1] != X_ET[i].shape[1]:
                    trend_upsampled = trend_upsampled[:, :X_ET[i].shape[1], :]
                X_ET[i] = X_ET[i] + trend_upsampled

        # 2.2 局部项：自下而上（细→粗）
        for i in range(1, m_plus_1):
            if i - 1 >= 0:
                # 传入目标长度（当前粗尺度的长度）
                target_len = X_EL[i].shape[1]
                local_downsampled = self.local_linears[i - 1](X_EL[i - 1], target_len)
                # 最终校验：确保长度完全一致
                if local_downsampled.shape[1] != X_EL[i].shape[1]:
                    local_downsampled = local_downsampled[:, :X_EL[i].shape[1], :]
                X_EL[i] = X_EL[i] + local_downsampled

        # 3. 线性一致化融合
        integrated_subs = []
        for i in range(m_plus_1):
            fusion = self.linear_fusion(X_ET[i] + X_EL[i])
            integrated = encoded_subs[i] + self.dropout(fusion)
            integrated_subs.append(integrated)

        return integrated_subs


class HIIM(nn.Module):
    """历史信息整合模块（两层HIIM，论文公式5-11~5-12）"""

    def __init__(self, m=3, f_enc=128, hidden_dim=256, kernel_n=[2, 4, 6]):
        super().__init__()
        self.hiim1 = HIIMBlock(m, f_enc, hidden_dim, kernel_n)
        self.hiim2 = HIIMBlock(m, f_enc, hidden_dim, kernel_n)

    def forward(self, encoded_subs):
        """
        :param encoded_subs: 编码后的子序列列表 X_E^0
        :return: 两层整合后的子序列列表 X_E^1, X_E^2
        """
        X_E1 = self.hiim1(encoded_subs)
        X_E2 = self.hiim2(X_E1)
        return X_E1, X_E2


# 独立调试代码
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 测试参数
    m = 3
    f_enc = 128
    hidden_dim = 256

    # 模拟MEDEM输出（编码后的子序列列表）
    encoded_subs = [
        torch.randn(16, 500, f_enc).to(device),  # X0^0
        torch.randn(16, 250, f_enc).to(device),  # X1^0
        torch.randn(16, 125, f_enc).to(device),  # X2^0
        torch.randn(16, 62, f_enc).to(device)    # X3^0
    ]
    print("输入编码子序列形状：")
    for i, sub in enumerate(encoded_subs):
        print(f"X_{i}^0: {sub.shape}")

    # 初始化HIIM
    hiim = HIIM(m=m, f_enc=f_enc, hidden_dim=hidden_dim).to(device)

    # 前向传播
    X_E1, X_E2 = hiim(encoded_subs)

    # 输出测试结果
    print("\nHIIM1输出（X_E^1）形状：")
    for i, sub in enumerate(X_E1):
        print(f"X_{i}^1: {sub.shape}")
    print("\nHIIM2输出（X_E^2）形状：")
    for i, sub in enumerate(X_E2):
        print(f"X_{i}^2: {sub.shape}")
    print("HIIM模块测试完成！")