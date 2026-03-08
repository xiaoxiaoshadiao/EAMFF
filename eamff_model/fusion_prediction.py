import torch
import torch.nn as nn


class FusionPrediction(nn.Module):
    """融合预测模块：加权融合三路专家分量，输出最终预测"""

    def __init__(self, in_features):
        super(FusionPrediction, self).__init__()
        self.in_features = in_features

        # 统一输出映射层（确保融合后维度一致）
        self.out_proj = nn.Linear(in_features, in_features)

    def forward(self, Xs, Xbg, Xr, alpha):
        """
        :param Xs: (batch, predict_step, in_features) 局部扰动分量
        :param Xbg: (batch, predict_step, in_features) 背景状态分量
        :param Xr: (batch, predict_step, in_features) 长期恢复分量
        :param alpha: (batch, 3) 权重（αs, αbg, αr）
        :return: X_hat (batch, predict_step, in_features) 最终预测
        """
        batch_size, predict_step, _ = Xs.shape

        # 1. 权重广播：(batch, 3) → (batch, predict_step, 3)
        alpha_expand = alpha.unsqueeze(1).repeat(1, predict_step, 1)  # (batch, predict_step, 3)

        # 2. 加权融合（公式6-4）
        X_hat = (
                alpha_expand[:, :, 0:1] * Xs +  # αs * Xs
                alpha_expand[:, :, 1:2] * Xbg +  # αbg * Xbg
                alpha_expand[:, :, 2:3] * Xr  # αr * Xr
        )

        # 3. 统一输出映射
        X_hat = self.out_proj(X_hat)

        return X_hat


# 独立调试代码
if __name__ == "__main__":
    print("=== 调试fusion_prediction.py ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    batch_size = 4
    predict_step = 10
    in_features = 6
    Xs = torch.randn(batch_size, predict_step, in_features).to(device)
    Xbg = torch.randn(batch_size, predict_step, in_features).to(device)
    Xr = torch.randn(batch_size, predict_step, in_features).to(device)
    alpha = torch.tensor([[0.6, 0.4, 0.0], [0.3, 0.3, 0.4], [0.0, 0.0, 1.0], [0.2, 0.5, 0.3]],
                         dtype=torch.float32).to(device)

    # 初始化融合模块
    fusion = FusionPrediction(in_features=6).to(device)

    # 前向传播
    X_hat = fusion(Xs, Xbg, Xr, alpha)
    print(f"Xs/Xbg/Xr形状: {Xs.shape}")
    print(f"权重形状: {alpha.shape}")
    print(f"最终预测形状: {X_hat.shape}")

    # 验证维度
    assert X_hat.shape == (batch_size, predict_step, in_features), "融合后维度错误"
    print("✅ 融合预测模块调试通过")