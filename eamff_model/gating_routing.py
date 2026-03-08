import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingRouting(nn.Module):
    """步长一致性路由模块：硬路由+软加权，生成自适应权重"""

    def __init__(self, hidden_dim=64, max_o=150):
        super(GatingRouting, self).__init__()
        self.max_o = max_o

        # 可学习权重函数（输入：归一化步长，输出：3路权重）
        self.weight_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # αs, αbg, αr
        )

    def forward(self, o_batch):
        """
        :param o_batch: (batch, 1) 每个样本的预测步长
        :return: alpha (batch, 3) 满足αs+αbg+αr=1，且符合硬路由规则
        """
        batch_size = o_batch.shape[0]

        # 1. 归一化步长到[0,1]
        o_norm = o_batch / self.max_o  # (batch, 1)

        # 2. 软权重预测
        alpha_soft = self.weight_net(o_norm)  # (batch, 3)
        alpha_soft = F.softmax(alpha_soft, dim=1)  # 满足非负+和为1

        # 3. 硬路由规则（公式6-7）
        alpha_hard = alpha_soft.clone()
        for i in range(batch_size):
            o = o_batch[i, 0].item()
            if o <= 10:
                # 短期：αr=0
                alpha_hard[i, 2] = 0.0
                # 重新归一化αs和αbg
                alpha_hard[i, :2] = alpha_hard[i, :2] / alpha_hard[i, :2].sum()
            elif o >= 100:
                # 长期：αs=αbg=0
                alpha_hard[i, :2] = 0.0
                alpha_hard[i, 2] = 1.0
            # 中期：保持软权重（无需处理）

        # 最终权重（硬路由约束后的软权重）
        alpha = alpha_hard
        return alpha


# 独立调试代码
if __name__ == "__main__":
    print("=== 调试gating_routing.py ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化路由模块
    gating = GatingRouting(hidden_dim=64, max_o=150).to(device)

    # 测试不同步长的权重生成
    test_os = torch.tensor([[5], [50], [120]], dtype=torch.float32).to(device)  # 短/中/长
    alphas = gating(test_os)

    print("测试步长与对应权重（αs, αbg, αr）：")
    for i, o in enumerate([5, 50, 120]):
        alpha = alphas[i].cpu().detach().numpy()
        print(f"步长O={o}: 权重={alpha.round(4)}, 和={alpha.sum():.4f}")

    # 验证硬路由规则
    assert alphas[0, 2].item() == 0.0, "短期步长αr应=0"
    assert alphas[2, 0].item() == 0.0 and alphas[2, 1].item() == 0.0, "长期步长αs/αbg应=0"
    assert all([abs(alpha.sum() - 1.0) < 1e-5 for alpha in alphas]), "权重和应=1"
    print("✅ 步长一致性路由规则验证通过")