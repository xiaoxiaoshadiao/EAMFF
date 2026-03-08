import torch
import torch.nn as nn
import sys
import os

# ===================== 关键修正：完善系统路径 =====================
current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_dir, "../"))
famm_dir = os.path.join(parent_dir, "famm_model")
ramm_dir = os.path.join(parent_dir, "ramm_model")

sys.path.extend([parent_dir, famm_dir, ramm_dir])
print(f"📌 系统路径已添加：")
print(f"   - 上级目录: {parent_dir}")
print(f"   - FAMM目录: {famm_dir}")
print(f"   - RAMM目录: {ramm_dir}")

# ===================== 导入真实FAMM/RAMM模块 =====================
try:
    from famm_core import FAMM as RealFAMM
    from ramm_core import RAMM as RealRAMM

    print("✅ 成功导入真实FAMM/RAMM核心模块（含依赖的multi_scale_decompose）")
except ImportError as e:
    print(f"⚠️  导入真实FAMM/RAMM失败: {e}")
    print("⚠️  自动降级为简化版FAMM/RAMM（仅用于调试）")


    # 降级方案：保留简化版，且确保有self.in_features + return_intermediate参数
    class RealFAMM(nn.Module):
        def __init__(self, in_features, d_model=64, predict_step=10):
            super(RealFAMM, self).__init__()
            self.in_features = in_features  # 确保有实例属性
            self.d_model = d_model
            self.predict_step = predict_step
            # 模拟多尺度分解
            self.msd = nn.Sequential(
                nn.Conv1d(in_features, in_features, kernel_size=3, padding=1),
                nn.ReLU()
            )
            # ICB模块（局部扰动Xs）
            self.icb = nn.Sequential(
                nn.Conv1d(in_features, d_model, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
            # GRU背景通道（Xbg）
            self.gru_bg = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
            # 投影层
            self.proj_s = nn.Linear(d_model, in_features)
            self.proj_bg = nn.Linear(d_model, in_features)

        def forward(self, x, return_intermediate=False):
            batch_size = x.shape[0]
            # 模拟多尺度分解
            x_trans = x.permute(0, 2, 1)
            xs = self.msd(x_trans).permute(0, 2, 1)
            xbg = self.msd(x_trans).permute(0, 2, 1)
            # 模拟预测结果
            pred_out = torch.randn(batch_size, self.predict_step, self.in_features).to(x.device)
            # 兼容return_intermediate参数
            if return_intermediate:
                return pred_out, xs, xbg
            else:
                return pred_out


    class RealRAMM(nn.Module):
        def __init__(self, in_features, f_enc=128, predict_step=10):
            super(RealRAMM, self).__init__()
            self.in_features = in_features  # 确保有实例属性
            self.f_enc = f_enc
            self.predict_step = predict_step
            # MEDEM特征编码
            self.medem = nn.Sequential(
                nn.Linear(in_features, f_enc),
                nn.ReLU(),
                nn.Linear(f_enc, f_enc),
                nn.ReLU()
            )
            # HIIM长期趋势建模
            self.hiim = nn.LSTM(f_enc, f_enc, num_layers=2, batch_first=True)
            # MPM输出映射
            self.mpm = nn.Linear(f_enc, in_features)

        def forward(self, x):
            x_enc = self.medem(x)
            lstm_out, _ = self.hiim(x_enc)
            xr_feat = lstm_out[:, -self.predict_step:, :]
            Xr = self.mpm(xr_feat)
            return Xr


# ===================== 专家封装模块 =====================
class ExpertEncapsulation(nn.Module):
    """专家分量封装模块：整合真实FAMM的Xs/Xbg和真实RAMM的Xr，统一维度"""

    def __init__(self, in_features, famm_d_model=64, ramm_f_enc=128, max_o=150):
        super(ExpertEncapsulation, self).__init__()
        self.in_features = in_features
        self.max_o = max_o

        # 初始化真实的FAMM和RAMM（支持动态预测步长）
        self.famm = RealFAMM(
            in_features=in_features,
            d_model=famm_d_model,
            predict_step=max_o  # FAMM的predict_step设为最大步长
        )
        self.ramm = RealRAMM(
            in_features=in_features,
            f_enc=ramm_f_enc,
            predict_step=max_o,
            m=3,  # 真实RAMM的默认参数
            hidden_dim=256,  # 真实RAMM的默认参数
            kernel_n=[2, 4, 6]  # 真实RAMM的默认参数
        )

        # 维度统一投影层（确保三路分量维度一致）
        famm_in_feat = getattr(self.famm, 'in_features', in_features)
        ramm_in_feat = getattr(self.ramm, 'in_features', in_features)
        self.unify_dim_s = nn.Linear(famm_in_feat, in_features)
        self.unify_dim_bg = nn.Linear(famm_in_feat, in_features)
        self.unify_dim_r = nn.Linear(ramm_in_feat, in_features)

    def forward(self, x, predict_step):
        """
        :param x: (batch, window_len, in_features) 输入序列
        :param predict_step: int 预测步长O
        :return: Xs, Xbg, Xr (均为batch, predict_step, in_features)
        """
        # 安全校验：预测步长不能超过最大支持步长
        if predict_step > self.max_o:
            raise ValueError(f"预测步长{predict_step}超过最大支持步长{self.max_o}")

        # ========== 核心修正：调用FAMM获取中间变量xs/xbg ==========
        # 调用FAMM并返回中间分解值（xs=局部，xbg=背景）
        _, xs_full, xbg_full = self.famm(x, return_intermediate=True)
        print(f"📝 FAMM分解结果 - xs形状: {xs_full.shape}, xbg形状: {xbg_full.shape}")

        # ========== 处理RAMM的输出 ==========
        xr_full = self.ramm(x)  # RAMM输出: (batch, max_o, in_features)
        print(f"📝 RAMM输出 - xr形状: {xr_full.shape}")

        # ========== 截断到目标预测步长 ==========
        # 注意：FAMM的xs/xbg是(window_len, in_features)，需截取最后max_o个时间步（匹配RAMM）
        xs_full = xs_full[:, -self.max_o:, :]  # 截取最后max_o个时间步
        xbg_full = xbg_full[:, -self.max_o:, :]

        # 再截断到目标predict_step
        Xs = self.unify_dim_s(xs_full[:, :predict_step, :])
        Xbg = self.unify_dim_bg(xbg_full[:, :predict_step, :])
        Xr = self.unify_dim_r(xr_full[:, :predict_step, :])

        # 最终校验：三路分量维度必须一致
        assert Xs.shape == Xbg.shape == Xr.shape, \
            f"专家分量维度不一致！Xs={Xs.shape}, Xbg={Xbg.shape}, Xr={Xr.shape}"
        return Xs, Xbg, Xr


# ===================== 独立调试代码 =====================
if __name__ == "__main__":
    print("\n=== 调试expert_encapsulation.py（真实导入版）===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模拟输入：batch=4, window_len=500, in_features=6
    x = torch.randn(4, 500, 6).to(device)
    predict_step = 10

    # 初始化专家封装模块
    expert = ExpertEncapsulation(
        in_features=6,
        famm_d_model=64,
        ramm_f_enc=128,
        max_o=150
    ).to(device)

    # 验证FAMM/RAMM的实例属性
    print(f"✅ FAMM.in_features: {expert.famm.in_features}")
    print(f"✅ RAMM.in_features: {expert.ramm.in_features}")

    # 前向传播验证
    try:
        Xs, Xbg, Xr = expert(x, predict_step)
        print(f"\n✅ 前向传播成功！")
        print(f"输入形状: {x.shape}")
        print(f"Xs形状 (局部扰动): {Xs.shape}")
        print(f"Xbg形状 (背景状态): {Xbg.shape}")
        print(f"Xr形状 (长期恢复): {Xr.shape}")
        print(f"✅ 专家分量维度一致，封装模块调试通过")
    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback

        traceback.print_exc()
        raise