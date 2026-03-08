import torch
import torch.nn as nn
import torch.nn.functional as F


class MPM(nn.Module):
    """融合预测模块（MPM，论文公式5-13~5-14）"""

    def __init__(self, m=3, f_enc=128, in_features=6, predict_step=100):
        super().__init__()
        self.m = m
        self.predict_step = predict_step
        self.in_features = in_features

        # 多尺度线性投影层（每个尺度1个）
        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(f_enc, f_enc),
                nn.ReLU(),
                nn.Linear(f_enc, in_features)
            ) for _ in range(m + 1)
        ])

    def forward(self, X_E1, X_E2):
        """
        :param X_E1: HIIM1输出列表
        :param X_E2: HIIM2输出列表
        :return: 最终预测结果 (batch, predict_step, in_features)
        """
        batch_size = X_E1[0].shape[0]
        total_pred = torch.zeros(batch_size, self.predict_step, self.in_features).to(X_E1[0].device)

        # 遍历每个尺度
        for i in range(len(X_E1)):
            # 1. 提取当前尺度的两层特征
            x1 = X_E1[i]
            x2 = X_E2[i]

            # 2. 线性投影
            y1 = self.scale_projectors[i](x1)
            y2 = self.scale_projectors[i](x2)
            y_m = y1 + y2  # 论文公式5-13

            # 3. 插值到预测步长
            y_m_interp = F.interpolate(
                y_m.transpose(1, 2),
                size=self.predict_step,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

            # 4. 累加到总预测结果（论文公式5-14）
            total_pred += y_m_interp

        return total_pred


# 独立调试代码
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试参数
    m = 3
    f_enc = 128
    in_features = 6
    predict_step = 100

    # 模拟HIIM输出
    X_E1 = [
        torch.randn(16, 500, f_enc).to(device),
        torch.randn(16, 250, f_enc).to(device),
        torch.randn(16, 125, f_enc).to(device),
        torch.randn(16, 62, f_enc).to(device)
    ]
    X_E2 = [
        torch.randn(16, 500, f_enc).to(device),
        torch.randn(16, 250, f_enc).to(device),
        torch.randn(16, 125, f_enc).to(device),
        torch.randn(16, 62, f_enc).to(device)
    ]

    # 初始化MPM
    mpm = MPM(m=m, f_enc=f_enc, in_features=in_features, predict_step=predict_step).to(device)

    # 前向传播
    pred = mpm(X_E1, X_E2)

    # 输出测试结果
    print(f"MPM输出形状: {pred.shape}")  # 预期 (16, 100, 6)
    print("MPM模块测试完成！")