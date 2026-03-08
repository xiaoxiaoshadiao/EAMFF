import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置全局随机种子
torch.manual_seed(42)
np.random.seed(42)


class GRUBackgroundModel(nn.Module):
    """
    GRU背景状态建模模块
    对应论文4.2.3节：建模背景项Xbg的慢变趋势+历史恢复信息
    """

    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.1):
        super(GRUBackgroundModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU层（双向=False，贴合论文单向时序建模）
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 投影层（保证输出维度与ICB特征一致）
        self.proj_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, xbg):
        """
        前向传播：处理背景项Xbg，输出背景特征Ybg
        :param xbg: 背景项 (batch, window_len, in_features)
        :return: ybg (batch, window_len, hidden_dim)
        """
        # GRU前向传播
        gru_out, _ = self.gru(xbg)  # (batch, L, hidden_dim)

        # 投影+归一化+正则化
        ybg = self.proj_linear(gru_out)
        ybg = self.layer_norm(ybg)
        ybg = self.dropout(ybg)

        return ybg


class GatedFusion(nn.Module):
    """
    门控融合模块
    对应论文4.2.3节：自适应加权局部特征Ys和背景特征Ybg
    """

    def __init__(self, d_model=64, out_features=6):
        super(GatedFusion, self).__init__()
        self.d_model = d_model

        # 权重计算层（拼接Ys和Ybg后计算α）
        self.gate_linear = nn.Linear(2 * d_model, 1)  # 输出单通道权重

        # 输出投影层（高维→原始特征维度）
        self.out_linear = nn.Linear(d_model, out_features)

    def forward(self, ys, ybg):
        """
        前向传播：融合局部特征和背景特征，输出预测序列
        :param ys: 局部特征 (batch, window_len, d_model)
        :param ybg: 背景特征 (batch, window_len, d_model)
        :return: y_pred (batch, window_len, out_features) → 预测序列
        """
        batch_size, window_len, _ = ys.shape

        # 1. 特征拼接 (batch, L, 2*d_model)
        concat_feat = torch.cat([ys, ybg], dim=-1)

        # 2. 计算门控权重α（Sigmoid→[0,1]）
        alpha = self.gate_linear(concat_feat)  # (batch, L, 1)
        alpha = torch.sigmoid(alpha)  # 逐时刻权重

        # 3. 自适应加权融合
        y_fusion = alpha * ys + (1 - alpha) * ybg  # (batch, L, d_model)

        # 4. 投影到原始特征维度
        y_pred = self.out_linear(y_fusion)  # (batch, L, out_features)

        # 输出权重（调试用）
        if self.training is False:
            alpha_mean = torch.mean(alpha).item()
            print(f"门控权重α均值: {alpha_mean:.3f} (预期在0~1之间)")

        return y_pred


# ==================== Debug测试 ====================
if __name__ == "__main__":
    print("=== 测试gated_fusion.py功能 ===")

    # 1. 构造测试数据
    batch_size = 8
    window_len = 300
    in_features = 6
    d_model = 64

    # 背景项Xbg（模拟多尺度分解输出）
    test_xbg = torch.randn(batch_size, window_len, in_features)
    # 局部特征Ys（模拟ICB输出）
    test_ys = torch.randn(batch_size, window_len, d_model)

    # 2. 测试GRU背景建模
    print("\n1. 测试GRU背景建模模块...")
    gru_bg = GRUBackgroundModel(in_features=in_features, hidden_dim=d_model)
    test_ybg = gru_bg(test_xbg)
    print(f"GRU输出Ybg形状: {test_ybg.shape}")
    assert test_ybg.shape == (batch_size, window_len, d_model), "Ybg维度不符合预期"

    # 3. 测试门控融合模块
    print("\n2. 测试门控融合模块...")
    gated_fusion = GatedFusion(d_model=d_model, out_features=in_features)
    gated_fusion.eval()  # 评估模式（输出权重均值）
    test_y_pred = gated_fusion(test_ys, test_ybg)
    print(f"融合输出预测序列形状: {test_y_pred.shape}")
    assert test_y_pred.shape == (batch_size, window_len, in_features), "预测序列维度不符合预期"

    print("\n=== gated_fusion.py测试完成，功能正常 ===")