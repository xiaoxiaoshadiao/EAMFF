import numpy as np
import torch
import torch.nn as nn
from multi_scale_decompose import MultiScaleDecompose
from isometric_conv_block import IsometricConvBlock
from gated_fusion import GRUBackgroundModel, GatedFusion

# 设置全局随机种子
torch.manual_seed(42)
np.random.seed(42)


class FAMM(nn.Module):
    """
    FAMM模型核心（组合所有模块）
    对应论文4.2节完整架构：多尺度分解→ICB→GRU→门控融合
    """

    def __init__(self, in_features=6, d_model=64, predict_step=1):
        super(FAMM, self).__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.predict_step = predict_step

        # 1. 多尺度分解模块
        self.msd = MultiScaleDecompose(kernel1=3, kernel2=5)

        # 2. 等距卷积模块（ICB）
        self.icb = IsometricConvBlock(in_features=in_features, d_model=d_model)

        # 3. GRU背景建模模块
        self.gru_bg = GRUBackgroundModel(in_features=in_features, hidden_dim=d_model)

        # 4. 门控融合模块
        self.gated_fusion = GatedFusion(d_model=d_model, out_features=in_features)

        # 5. 预测头（截取最后predict_step个时间步作为输出）
        self.predict_head = nn.Linear(in_features, in_features)  # 线性投影（保持维度）

    def forward(self, x):
        """
        前向传播：完整FAMM模型流程
        :param x: 输入序列 (batch, window_len, in_features)
        :return: pred_out (batch, predict_step, in_features) → 最终预测结果
        """
        # 1. 多尺度分解：X → Xs（局部） + Xbg（背景）
        xs, xbg = self.msd(x)

        # 2. ICB处理局部项：Xs → Ys
        ys = self.icb(xs)

        # 3. GRU处理背景项：Xbg → Ybg
        ybg = self.gru_bg(xbg)

        # 4. 门控融合：Ys + Ybg → 融合特征
        fusion_out = self.gated_fusion(ys, ybg)

        # 5. 预测头：截取最后predict_step个时间步作为输出
        pred_out = fusion_out[:, -self.predict_step:, :]  # (batch, predict_step, in_features)
        pred_out = self.predict_head(pred_out)

        return pred_out


# ==================== Debug测试 ====================
if __name__ == "__main__":
    print("=== 测试famm_core.py功能 ===")

    # 1. 构造测试数据（模拟预处理后的训练数据）
    batch_size = 8
    window_len = 300
    in_features = 6
    predict_step = 1
    test_x = torch.randn(batch_size, window_len, in_features)
    print(f"模型输入形状: {test_x.shape}")

    # 2. 初始化FAMM模型
    famm_model = FAMM(in_features=in_features, d_model=64, predict_step=predict_step)
    famm_model.eval()  # 评估模式

    # 3. 前向传播
    pred_out = famm_model(test_x)
    print(f"模型输出形状 (batch, predict_step, features): {pred_out.shape}")

    # 验证输出维度
    assert pred_out.shape == (batch_size, predict_step, in_features), "模型输出维度不符合预期"

    print("\n=== famm_core.py测试完成，FAMM模型核心功能正常 ===")