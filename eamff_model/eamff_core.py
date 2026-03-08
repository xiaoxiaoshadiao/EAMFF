import torch
import torch.nn as nn
import os
import sys

# ===================== 导入专家封装模块 =====================
# 确保能找到expert_encapsulation.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from expert_encapsulation import ExpertEncapsulation  # 导入你写的封装类


class EAMFF(nn.Module):
    """EAMFF主模型：集成ExpertEncapsulation，新增expert_encapsulation方法"""

    def __init__(self, in_features, famm_d_model=64, ramm_f_enc=128,
                 gating_hidden_dim=64, max_o=150):
        super(EAMFF, self).__init__()
        self.in_features = in_features
        self.max_o = max_o

        # ========== 核心：实例化专家封装模块 ==========
        self.expert_encap = ExpertEncapsulation(
            in_features=in_features,
            famm_d_model=famm_d_model,
            ramm_f_enc=ramm_f_enc,
            max_o=max_o
        )

        # 门控权重网络（和你原有的逻辑保持一致）
        self.gating_net = nn.Sequential(
            nn.Linear(1, gating_hidden_dim),  # 输入是预测步长o
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, 1),
            nn.Sigmoid()  # 输出α∈[0,1]
        )

        # 融合层（融合Xs/Xbg/Xr）
        self.fusion_layer = nn.Linear(in_features * 3, in_features)

    def expert_encapsulation(self, x, predict_step):
        """
        可视化函数调用的核心方法（名字必须和可视化里的一致！）
        :param x: 输入序列 (batch, window_len, in_features)
        :param predict_step: 预测步长
        :return: Xs, Xbg, Xr 真实分量
        """
        # 调用封装模块的forward，返回真实分量
        Xs, Xbg, Xr = self.expert_encap(x, predict_step)
        return Xs, Xbg, Xr

    def gating_weight_fn(self, o):
        """门控权重计算（和你原有的逻辑保持一致）"""
        # o是预测步长，形状：(batch,) 或 标量
        if isinstance(o, int):
            o_tensor = torch.tensor([[o]], dtype=torch.float32).to(next(self.parameters()).device)
        else:
            o_tensor = o.unsqueeze(1) if len(o.shape) == 1 else o
        alpha = self.gating_net(o_tensor)
        return alpha

    def forward(self, x, predict_step):
        """主前向传播：修改为返回5个值，匹配train_predict_eamff的解包逻辑"""
        # 1. 获取专家分量
        Xs, Xbg, Xr = self.expert_encapsulation(x, predict_step)

        # 2. 计算门控权重α
        alpha = self.gating_weight_fn(predict_step)

        # 3. 融合分量（示例逻辑，可按你的需求改）
        fused = torch.cat([Xs, Xbg, Xr], dim=-1)
        X_hat = self.fusion_layer(fused)  # 预测输出（改名为X_hat，匹配train_predict_eamff）

        # ========== 核心修改：返回5个值 ==========
        # 顺序：X_hat(预测输出), alpha(门控权重), Xs, Xbg, Xr
        return X_hat, alpha, Xs, Xbg, Xr


# ===================== 独立调试 =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模拟输入：batch=4, window_len=500, in_features=6
    x = torch.randn(4, 500, 6).to(device)
    predict_step = 10

    # 初始化EAMFF
    eamff = EAMFF(
        in_features=6,
        famm_d_model=64,
        ramm_f_enc=128,
        gating_hidden_dim=64,
        max_o=150
    ).to(device)

    # 测试forward返回值（匹配train_predict_eamff的解包）
    try:
        X_hat, alpha, Xs, Xbg, Xr = eamff(x, predict_step)
        print(f"✅ EAMFF.forward 返回5个值成功！")
        print(f"X_hat形状: {X_hat.shape}, alpha形状: {alpha.shape}")
        print(f"Xs形状: {Xs.shape}, Xbg形状: {Xbg.shape}, Xr形状: {Xr.shape}")
    except Exception as e:
        print(f"❌ forward调用失败: {e}")
        import traceback

        traceback.print_exc()