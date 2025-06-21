import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP库冲突

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# 尝试设置中文字体
def set_chinese_font():
    """尝试设置可用的中文字体"""
    # 常用中文字体列表
    chinese_fonts = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC",
                     "Microsoft YaHei", "SimSun", "WenQuanYi Micro Hei"]

    # 检查系统中可用的字体
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    # 找到第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            print(f"已设置中文字体: {font}")
            return True

    # 如果没有找到中文字体，使用默认字体
    print("警告: 未找到可用的中文字体，图表中的中文可能无法正确显示")
    return False


# 设置中文字体
set_chinese_font()

# 确保负号正确显示
plt.rcParams["axes.unicode_minus"] = False


# 理论公式计算函数
def theoretical_M_DC(f_n, Q):
    """
    根据理论公式计算 M_DC 值
    公式: M_DC(f_n, Q) ≈ 0.94049 / sqrt( (6/5 - 1/(5f_n²))² + Q²*(-1/f_n + f_n)² )
    """
    # 避免除以零
    f_n = np.clip(f_n, 0.1, None)

    term1 = (6 / 5 - 1 / (5 * f_n ** 2))
    term2 = Q * (f_n - 1 / f_n)  # 等价于 Q*(-1/f_n + f_n)
    denominator = np.sqrt(term1 ** 2 + term2 ** 2)

    # 避免分母为零
    denominator = np.where(denominator == 0, 1e-10, denominator)

    return 0.94049 / denominator


# 模型定义类，与训练时保持一致
class M_DC_Model(torch.nn.Module):
    """用于拟合M_DC(f_n, Q)的神经网络模型"""

    def __init__(self, input_size=2, hidden_size=256, output_size=1,
                 num_hidden_layers=4, activation="leaky_relu"):
        super(M_DC_Model, self).__init__()

        # 输入层
        layers = [torch.nn.Linear(input_size, hidden_size)]

        # 添加激活函数
        layers.append(self.get_activation(activation))

        # 隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(self.get_activation(activation))

        # 输出层
        layers.append(torch.nn.Linear(hidden_size, output_size))

        self.model = torch.nn.Sequential(*layers)

    def get_activation(self, name):
        """根据名称返回激活函数"""
        if name == "relu":
            return torch.nn.ReLU()
        elif name == "leaky_relu":
            return torch.nn.LeakyReLU(0.01)
        elif name == "tanh":
            return torch.nn.Tanh()
        elif name == "sigmoid":
            return torch.nn.Sigmoid()
        elif name == "elu":
            return torch.nn.ELU()
        else:
            raise ValueError(f"未知激活函数: {name}")

    def forward(self, x):
        return self.model(x)


def main():
    # 模型路径（与训练时设置一致）
    model_path = "results/M_DC_model.pth"

    # 1. 加载模型
    print(f"正在加载模型: {model_path}")
    model = M_DC_Model(hidden_size=256, num_hidden_layers=4, activation="leaky_relu")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("模型加载完成")

    # 2. 生成fn和Q的网格数据
    print("正在生成数据网格...")
    fn_range = np.linspace(0.1, 2.0, 100)  # fn范围
    Q_range = np.linspace(0.1, 10.0, 100)  # Q范围

    # 创建网格
    FN, Q = np.meshgrid(fn_range, Q_range)

    # 3. 使用模型预测M_DC值
    print("正在使用模型预测M_DC值...")
    M_DC = np.zeros_like(FN)

    # 批量预测以提高效率
    batch_size = 1000
    for i in range(len(FN)):
        for j in range(0, len(FN[i]), batch_size):
            end_idx = min(j + batch_size, len(FN[i]))
            # 准备输入数据
            inputs = torch.tensor(
                np.column_stack((FN[i, j:end_idx].flatten(), Q[i, j:end_idx].flatten())),
                dtype=torch.float32
            )
            # 预测
            with torch.no_grad():
                outputs = model(inputs).numpy()

            # 确保输出是一维数组
            M_DC[i, j:end_idx] = outputs.flatten()

    # 4. 绘制三维曲面图
    print("正在绘制三维图像...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    surf = ax.plot_surface(FN, Q, M_DC, cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.8)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='M_DC值')

    # 设置坐标轴标签和标题
    ax.set_xlabel('f_n', fontsize=12)
    ax.set_ylabel('Q', fontsize=12)
    ax.set_zlabel('M_DC', fontsize=12)
    ax.set_title('神经网络拟合的M_DC(f_n, Q)三维曲面图', fontsize=14)

    # 设置视角
    ax.view_init(elev=30, azim=45)  # 调整视角

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.5)

    # 优化布局
    plt.tight_layout()

    # 保存图像
    plot_path = "results/M_DC_3D_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"三维图像已保存至: {plot_path}")

    # 5. 绘制特定Q值下理论值与神经网络输出对比曲线
    print("正在绘制特定Q值下理论值与神经网络输出对比曲线...")
    q_values = [0.25, 0.32, 0.4, 0.6, 1, 2, 4, 10]
    colors = plt.cm.viridis(np.linspace(0, 1, len(q_values)))

    plt.figure(figsize=(12, 8))

    for i, q in enumerate(q_values):
        # 计算理论值
        theory_curve = [theoretical_M_DC(f_n, q) for f_n in fn_range]

        # 准备模型输入
        inputs = torch.tensor(
            np.column_stack((fn_range, np.full_like(fn_range, q))),
            dtype=torch.float32
        )

        # 模型预测
        with torch.no_grad():
            preds = model(inputs).numpy().flatten()

        # 绘制理论曲线（虚线）
        plt.plot(fn_range, theory_curve, '--',
                 color=colors[i],
                 linewidth=2.5,
                 label=f'理论公式 (Q={q})')

        # 绘制模型预测曲线（实线）
        plt.plot(fn_range, preds, '-',
                 color=colors[i],
                 linewidth=1.5,
                 label=f'神经网络 (Q={q})')

    # 添加标记和标签
    plt.xlabel('f_n', fontsize=12)
    plt.ylabel('M_DC', fontsize=12)
    plt.title('不同Q值下理论公式与神经网络输出对比', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整图例位置（向上移动）
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               ncol=2, fontsize=10, framealpha=0.9)


    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 调整底部边距

    # 保存图像
    comparison_path = "results/M_DC_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"对比曲线图像已保存至: {comparison_path}")

    # 6. 绘制理论值和神经网络误差的三维图（带正负号）
    print("正在绘制理论值和神经网络误差的三维图...")
    # 计算理论值网格
    M_DC_theory = np.zeros_like(FN)
    for i in range(len(FN)):
        for j in range(len(FN[i])):
            M_DC_theory[i, j] = theoretical_M_DC(FN[i, j], Q[i, j])

    # 计算带正负号的误差
    error = M_DC - M_DC_theory  # 直接计算差值，保留正负号

    # 创建误差三维图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 使用蓝白红颜色映射，蓝色表示负误差，红色表示正误差
    error_surf = ax.plot_surface(FN, Q, error, cmap=cm.RdBu,
                                 linewidth=0, antialiased=True, alpha=0.8)

    # 添加颜色条
    fig.colorbar(error_surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='误差值')

    # 设置坐标轴标签和标题
    ax.set_xlabel('f_n', fontsize=12)
    ax.set_ylabel('Q', fontsize=12)
    ax.set_zlabel('误差值', fontsize=12)
    ax.set_title('理论值与神经网络预测的误差三维图', fontsize=14)

    # 设置视角
    ax.view_init(elev=30, azim=45)  # 调整视角

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.5)

    # 优化布局
    plt.tight_layout()

    # 保存图像
    error_path = "results/M_DC_error_3D.png"
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    print(f"误差三维图像已保存至: {error_path}")

    # 显示所有图像
    plt.show()


if __name__ == "__main__":
    main()