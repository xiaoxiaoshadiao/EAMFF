import torch
import torch.nn as nn
from medem import MEDEM
from hiim import HIIM
from mpm import MPM

class RAMM(nn.Module):
    """RAMM核心模型（MEDEM + HIIM + MPM）"""
    def __init__(self, in_features=6, m=3, f_enc=128, hidden_dim=256, kernel_n=[2, 4, 6], predict_step=100):
        super().__init__()
        # ========== 关键修正：添加实例属性 ==========
        self.in_features = in_features  # 必须赋值，否则外部无法访问
        self.m = m
        self.f_enc = f_enc
        self.hidden_dim = hidden_dim
        self.kernel_n = kernel_n
        self.predict_step = predict_step

        # 1. 多尺度指数分解编码模块
        self.medem = MEDEM(in_features=in_features, m=m, f_enc=f_enc)
        # 2. 历史信息整合模块
        self.hiim = HIIM(m=m, f_enc=f_enc, hidden_dim=hidden_dim, kernel_n=kernel_n)
        # 3. 融合预测模块
        self.mpm = MPM(m=m, f_enc=f_enc, in_features=in_features, predict_step=predict_step)

    def forward(self, x):
        """
        :param x: 输入张量 (batch, window_len, in_features)
        :return: 预测结果 (batch, predict_step, in_features)
        """
        # 1. MEDEM：指数分解+编码
        encoded_subs = self.medem(x)
        # 2. HIIM：两层历史信息整合
        X_E1, X_E2 = self.hiim(encoded_subs)
        # 3. MPM：多尺度融合预测
        pred = self.mpm(X_E1, X_E2)
        return pred

# 独立调试代码
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试参数
    batch_size = 16
    window_len = 500
    in_features = 6
    m = 3
    f_enc = 128
    predict_step = 100

    # 生成测试数据
    x = torch.randn(batch_size, window_len, in_features).to(device)
    print(f"输入形状: {x.shape}")

    # 初始化RAMM模型
    ramm = RAMM(
        in_features=in_features,
        m=m,
        f_enc=f_enc,
        predict_step=predict_step
    ).to(device)

    # 验证实例属性是否存在
    print(f"RAMM.in_features: {ramm.in_features}")  # 新增验证
    # 前向传播
    pred = ramm(x)

    # 输出测试结果
    print(f"RAMM预测输出形状: {pred.shape}")  # 预期 (16, 100, 6)
    print("RAMM核心模型测试完成！")