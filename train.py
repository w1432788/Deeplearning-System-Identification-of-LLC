import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置使用支持数学符号的字体
mpl.rcParams['font.family'] = 'DejaVu Sans'  # 使用支持数学符号的字体
mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号
mpl.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX数学字体

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import warnings
import matplotlib.colors as mcolors
import joblib

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 简化警告过滤
warnings.filterwarnings("ignore", category=UserWarning)

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


# ===================== 用户可配置参数 =====================
CONFIG = {
    "data_path": "M_DC_dataset.csv",  # 数据集路径
    "test_size": 0.2,  # 测试集比例
    "batch_size": 1024,  # 批量大小
    "hidden_size": 256,  # 隐藏层神经元数量
    "num_hidden_layers": 4,  # 隐藏层数量
    "activation": "leaky_relu",  # 激活函数类型: relu, leaky_relu, tanh, sigmoid, elu
    "learning_rate": 0.0001,  # 初始学习率
    "num_epochs": 500,  # 最大训练轮数
    "early_stopping_patience": 40,  # 早停耐心值(epochs)
    "min_delta": 0.00001,  # 早停最小改善阈值
    "lr_scheduler_factor": 0.5,  # 学习率衰减因子
    "lr_scheduler_patience": 10,  # 学习率调度耐心值(epochs)
    "use_gpu": True,  # 是否使用GPU
    "save_model": True,  # 是否保存模型
    "model_path": "M_DC_model.pth",  # 模型保存路径
    "scaler_path": "scaler_params.pkl",  # 标准化参数保存路径
    "loss_plot_path": "training_loss.png",  # 损失曲线保存路径
    "comparison_plot_path": "comparison.png",  # 对比图保存路径
    "q_comparison_path": "q_comparison.png"  # Q值对比图保存路径
}


# ===================== 理论公式计算函数 =====================
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


# ===================== 神经网络模型定义 =====================
class M_DC_Model(nn.Module):
    """用于拟合M_DC(f_n, Q)的神经网络模型"""

    def __init__(self, input_size=2, hidden_size=256, output_size=1,
             num_hidden_layers=4, activation="leaky_relu"):
        super(M_DC_Model, self).__init__()

        # 输入层
        layers = [nn.Linear(input_size, hidden_size)]

        # 添加激活函数
        layers.append(self.get_activation(activation))

        # 隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.get_activation(activation))

        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def get_activation(self, name):
        """根据名称返回激活函数"""
        if name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.01)
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"未知激活函数: {name}")

    def forward(self, x):
        return self.model(x)


# ===================== 数据加载与预处理 =====================
def load_and_prepare_data(config):
    """加载并预处理数据集"""
    # 加载数据集
    print(f"正在加载数据集: {config['data_path']}")
    try:
        data = pd.read_csv(config["data_path"])
    except FileNotFoundError:
        print(f"错误: 找不到文件 {config['data_path']}")
        print("请确保文件存在或检查路径是否正确")
        exit(1)

    # 检查数据列
    if 'f_n' not in data.columns or 'Q' not in data.columns or 'M_DC' not in data.columns:
        print("错误: 数据集缺少必需的列 (f_n, Q, M_DC)")
        exit(1)

    X = data[['f_n', 'Q']].values
    y = data['M_DC'].values.reshape(-1, 1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=42
    )

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    print(f"数据集加载完成: 训练样本 {len(train_loader.dataset)}, 测试样本 {len(test_loader.dataset)}")
    return train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


# ===================== 训练函数 =====================
def train_model(model, train_loader, config, test_loader=None):
    """训练神经网络模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["lr_scheduler_factor"],
        patience=config["lr_scheduler_patience"]
    )

    # 早停机制
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # 记录训练过程
    train_losses = []
    test_losses = []
    learning_rates = []

    print(f"\n开始训练，使用设备: {device}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset) if test_loader else 0}")
    print(f"激活函数: {config['activation']}")
    print(f"初始学习率: {config['learning_rate']}")
    print(f"早停耐心值: {config['early_stopping_patience']} epochs")
    print(f"学习率调度器: 衰减因子 {config['lr_scheduler_factor']}, 耐心值 {config['lr_scheduler_patience']} epochs")
    print("-" * 60)

    # 训练开始时间
    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        # 训练批次
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # 计算平均训练损失
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 计算测试损失
        if test_loader:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item() * inputs.size(0)

            epoch_test_loss = test_loss / len(test_loader.dataset)
            test_losses.append(epoch_test_loss)
        else:
            epoch_test_loss = float('inf')

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 更新学习率调度器
        if test_loader:
            scheduler.step(epoch_test_loss)

        # 每5个epoch打印一次进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            epoch_time = time.time() - epoch_start_time
            test_info = f", 测试损失: {epoch_test_loss:.6f}" if test_loader else ""
            print(f"轮次 [{epoch + 1}/{config['num_epochs']}] | "
                  f"用时: {epoch_time:.2f}s | "
                  f"学习率: {current_lr:.6f} | "
                  f"训练损失: {epoch_train_loss:.6f}{test_info}")

        # 早停机制检查
        if epoch_test_loss < best_loss - config["min_delta"]:
            best_loss = epoch_test_loss
            epochs_no_improve = 0
            # 保存最佳模型
            if config["save_model"]:
                torch.save(model.state_dict(), config["model_path"])
                if epoch > 0:  # 避免第一次就打印
                    print(f"模型改进! 保存至 {config['model_path']}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["early_stopping_patience"]:
                print(f"早停! 连续 {epochs_no_improve} 轮无显著改进.")
                early_stop = True
                break

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {total_time:.2f}秒")

    # 如果没有早停，保存最终模型
    if config["save_model"] and not early_stop:
        torch.save(model.state_dict(), config["model_path"])
        print(f"模型已保存至 {config['model_path']}")

    return train_losses, test_losses, learning_rates


# ===================== 可视化函数 =====================
def plot_losses(train_losses, test_losses, config):
    """绘制训练和测试损失曲线"""
    plt.figure(figsize=(12, 7))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    if test_losses:
        plt.plot(epochs, test_losses, 'r-', label='测试损失')

    # 标记最低测试损失点
    if test_losses:
        min_loss_idx = np.argmin(test_losses)
        min_loss = test_losses[min_loss_idx]
        plt.scatter(min_loss_idx + 1, min_loss, c='red', s=100, label=f'最低测试损失: {min_loss:.6f}')
        plt.annotate(f'最低点: {min_loss:.4f}',
                     (min_loss_idx + 1, min_loss),
                     textcoords="offset points",
                     xytext=(0, 15),
                     ha='center')

    plt.xlabel('训练轮次')
    plt.ylabel('损失 (MSE)')
    plt.title('训练过程损失变化')
    plt.yscale('log')  # 对数尺度更好地显示变化
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(config["loss_plot_path"])
    print(f"损失曲线已保存至 {config['loss_plot_path']}")
    plt.close()


def plot_learning_rate(learning_rates, config):
    """绘制学习率变化曲线"""
    plt.figure(figsize=(12, 7))
    epochs = range(1, len(learning_rates) + 1)

    plt.plot(epochs, learning_rates, 'g-')
    plt.xlabel('训练轮次')
    plt.ylabel('学习率')
    plt.title('训练过程学习率变化')
    plt.yscale('log')  # 对数尺度更好地显示变化
    plt.grid(True)

    # 保存图像
    lr_plot_path = "learning_rate_history.png"
    plt.savefig(lr_plot_path)
    print(f"学习率变化曲线已保存至 {lr_plot_path}")
    plt.close()


def plot_comparison(model, X_test, y_test, config):
    """绘制模型预测与理论公式对比图"""
    device = next(model.parameters()).device
    model.eval()

    # 选择部分样本进行可视化以避免内存问题
    num_samples = min(1000, len(X_test))
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_test_sample = X_test[indices]
    y_test_sample = y_test[indices]

    with torch.no_grad():
        # 使用模型进行预测
        test_preds = model(X_test_sample.to(device)).cpu().numpy()

    # 计算理论值（使用原始值）
    X_test_numpy = X_test_sample.numpy()
    theoretical_values = np.array([
        theoretical_M_DC(f_n, Q) for f_n, Q in X_test_numpy
    ])

    # 创建对比图
    fig = plt.figure(figsize=(16, 12))

    # 1. 预测值 vs 真实值
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(y_test_sample, test_preds, alpha=0.5, c='blue')
    min_val = min(y_test_sample.min(), test_preds.min())
    max_val = max(y_test_sample.max(), test_preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('模型预测值')
    ax1.set_title('模型预测 vs 真实值')
    ax1.grid(True)

    # 添加R²分数
    r2 = r2_score(y_test_sample, test_preds)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. 模型预测 vs 理论公式
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(theoretical_values, test_preds, alpha=0.5, c='green')
    min_val = min(theoretical_values.min(), test_preds.min())
    max_val = max(theoretical_values.max(), test_preds.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('理论公式计算值')
    ax2.set_ylabel('模型预测值')
    ax2.set_title('模型预测 vs 理论公式')
    ax2.grid(True)

    # 添加R²分数
    r2_theory = r2_score(theoretical_values, test_preds)
    ax2.text(0.05, 0.95, f'R² = {r2_theory:.4f}', transform=ax2.transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. 固定Q，变化f_n的对比 (调整范围：f_n 0.1-2)
    ax3 = fig.add_subplot(2, 2, 3)
    Q_values = [0.5, 2, 10]  # 选择几个不同的Q值
    f_n_range = np.linspace(0.1, 2, 100)  # 调整范围

    for i, Q_val in enumerate(Q_values):
        # 理论曲线
        theory_curve = [theoretical_M_DC(f_n, Q_val) for f_n in f_n_range]
        ax3.plot(f_n_range, theory_curve, '--', lw=2, color=plt.cm.viridis(i / len(Q_values)),
                 label=f'理论 Q={Q_val}')

        # 模型预测曲线
        inputs = torch.tensor(np.column_stack((f_n_range, np.full(100, Q_val))),
                              dtype=torch.float32)
        with torch.no_grad():
            preds = model(inputs.to(device)).cpu().numpy().flatten()
        ax3.plot(f_n_range, preds, '-', lw=1.5, color=plt.cm.viridis(i / len(Q_values)),
                 label=f'模型 Q={Q_val}')

    ax3.set_xlabel('f_n')
    ax3.set_ylabel('M_DC')
    ax3.set_title('固定Q，变化f_n的对比 (f_n: 0.1-2)')
    ax3.legend()
    ax3.grid(True)

    # 4. 固定f_n，变化Q的对比 (调整范围：Q 0.1-10)
    ax4 = fig.add_subplot(2, 2, 4)
    f_n_values = [0.5, 1.0, 2.0]  # 选择几个不同的f_n值
    Q_range = np.linspace(0.1, 10, 100)  # 调整范围

    for i, f_n_val in enumerate(f_n_values):
        # 理论曲线
        theory_curve = [theoretical_M_DC(f_n_val, Q) for Q in Q_range]
        ax4.plot(Q_range, theory_curve, '--', lw=2, color=plt.cm.plasma(i / len(f_n_values)),
                 label=f'理论 f_n={f_n_val}')

        # 模型预测曲线
        inputs = torch.tensor(np.column_stack((np.full(100, f_n_val), Q_range)),
                              dtype=torch.float32)
        with torch.no_grad():
            preds = model(inputs.to(device)).cpu().numpy().flatten()
        ax4.plot(Q_range, preds, '-', lw=1.5, color=plt.cm.plasma(i / len(f_n_values)),
                 label=f'模型 f_n={f_n_val}')

    ax4.set_xlabel('Q')
    ax4.set_ylabel('M_DC')
    ax4.set_title('固定f_n，变化Q的对比 (Q: 0.1-10)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    # 保存图像
    plt.savefig(config["comparison_plot_path"])
    print(f"对比图已保存至 {config['comparison_plot_path']}")
    plt.close()


def plot_q_comparison(model, config):
    """
    绘制特定Q值下公式计算和神经网络输出的曲线对比
    Q值: [0.25, 0.32, 0.4, 0.6, 1, 2, 4, 10]
    f_n范围: 0.1-2
    """


    device = next(model.parameters()).device
    model.eval()

    # 定义要绘制的Q值
    q_values = [0.25, 0.32, 0.4, 0.6, 1, 2, 4, 10]

    # 创建颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(q_values)))

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 设置f_n范围 (0.1-2)
    f_n_range = np.linspace(0.1, 2, 200)

    # 为每个Q值绘制曲线
    for i, q in enumerate(q_values):
        # 计算理论值
        theory_curve = [theoretical_M_DC(f_n, q) for f_n in f_n_range]

        # 准备模型输入
        inputs = torch.tensor(np.column_stack((f_n_range, np.full_like(f_n_range, q))),
                              dtype=torch.float32)

        # 模型预测
        with torch.no_grad():
            preds = model(inputs.to(device)).cpu().numpy().flatten()

        # 绘制理论曲线（虚线）
        plt.plot(f_n_range, theory_curve, '--',
                 color=colors[i],
                 linewidth=2.5,
                 label=f'理论公式 (Q={q})')

        # 绘制模型预测曲线（实线）
        plt.plot(f_n_range, preds, '-',
                 color=colors[i],
                 linewidth=1.5,
                 label=f'神经网络 (Q={q})')

    # 添加标记和标签
    plt.xlabel('f_n (0.1-2)', fontsize=12)
    plt.ylabel('M_DC', fontsize=12)
    plt.title('不同Q值下公式计算与神经网络输出对比', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加图例（放在图形外部）
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, fontsize=10, framealpha=0.9)

    # 添加公式文本
    #plt.figtext(0.5, 0.01,
    #            r'$M_{DC}(f_n, Q) \approx \frac{0.94049}{\sqrt{\left( \frac{6}{5} - \frac{1}{5 f_n^2} \right)^2 + Q^2 \left( -\frac{1}{f_n} + f_n \right)^2}}$',
    #            ha='center', fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # 为图例和公式留出空间

    # 保存图像
    plt.savefig(config["q_comparison_path"], dpi=300, bbox_inches='tight')
    print(f"Q值对比图已保存至 {config['q_comparison_path']}")
    plt.close()

# def plot_q_comparison(model, config):
#     """
#     绘制特定Q值下公式计算和神经网络输出的曲线对比
#     Q值: [0.25, 0.32, 0.4, 0.6, 1, 2, 4, 10]
#     f_n范围: 0.1-2
#     """
#
#     device = next(model.parameters()).device
#     model.eval()
#
#     # 定义要绘制的Q值
#     q_values = [0.25, 0.32, 0.4, 0.6, 1, 2, 4, 10]
#
#     # 创建颜色映射
#     colors = plt.cm.viridis(np.linspace(0, 1, len(q_values)))
#
#     # 创建图形
#     plt.figure(figsize=(12, 8))
#
#     # 设置f_n范围 (0.1-2)
#     f_n_range = np.linspace(0.1, 2, 200)
#
#     # MATLAB中计算的标准化参数
#     f_n_mean = 1.0490
#     f_n_std = 0.5478
#     Q_mean = 5.0632
#     Q_std = 2.8580
#     M_DC_mean = 0.3507
#     M_DC_std = 0.3047
#
#     # 为每个Q值绘制曲线
#     for i, q in enumerate(q_values):
#         # 计算理论值（无需标准化）
#         theory_curve = [theoretical_M_DC(f_n, q) for f_n in f_n_range]
#
#         # 准备模型输入（使用相同的标准化参数）
#         f_n_scaled = (f_n_range - f_n_mean) / f_n_std
#         q_scaled = (q - Q_mean) / Q_std
#
#         inputs = torch.tensor(np.column_stack((f_n_scaled, np.full_like(f_n_scaled, q_scaled))),
#                               dtype=torch.float32)
#
#         # 模型预测
#         with torch.no_grad():
#             preds_scaled = model(inputs.to(device)).cpu().numpy().flatten()
#
#         # 反标准化模型输出
#         preds = preds_scaled * M_DC_std + M_DC_mean
#
#         # 绘制理论曲线（虚线）
#         plt.plot(f_n_range, theory_curve, '--',
#                  color=colors[i],
#                  linewidth=2.5,
#                  label=f'理论公式 (Q={q})')
#
#         # 绘制模型预测曲线（实线）
#         plt.plot(f_n_range, preds, '-',
#                  color=colors[i],
#                  linewidth=1.5,
#                  label=f'神经网络 (Q={q})')
#
#     # 添加标记和标签
#     plt.xlabel('f_n (0.1-2)', fontsize=12)
#     plt.ylabel('M_DC', fontsize=12)
#     plt.title('不同Q值下公式计算与神经网络输出对比', fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.7)
#
#     # 添加图例（放在图形外部）
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#                ncol=2, fontsize=10, framealpha=0.9)
#
#     # 添加公式文本
#     plt.figtext(0.5, 0.01,
#                 r'$M_{DC}(f_n, Q) \approx \frac{0.94049}{\sqrt{\left( \frac{6}{5} - \frac{1}{5 f_n^2} \right)^2 + Q^2 \left( -\frac{1}{f_n} + f_n \right)^2}}$',
#                 ha='center', fontsize=14)
#
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.25)  # 为图例和公式留出空间
#
#     # 保存图像
#     plt.savefig(config["q_comparison_path"], dpi=300, bbox_inches='tight')
#     print(f"Q值对比图已保存至 {config['q_comparison_path']}")
#     plt.close()

# ===================== 主程序 =====================
def main():
    # 创建输出目录
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # 更新输出路径
    config = CONFIG.copy()
    config["loss_plot_path"] = os.path.join(output_dir, config["loss_plot_path"])
    config["comparison_plot_path"] = os.path.join(output_dir, config["comparison_plot_path"])
    config["model_path"] = os.path.join(output_dir, config["model_path"])
    config["q_comparison_path"] = os.path.join(output_dir, "q_comparison.png")
    config["scaler_path"] = os.path.join(output_dir, config["scaler_path"])

    # 加载数据
    train_loader, test_loader, X_train, y_train, X_test, y_test = load_and_prepare_data(config)

    # 初始化模型
    model = M_DC_Model(
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        activation=config["activation"]
    )

    # 打印模型结构
    print("\n神经网络结构:")
    print(model)
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 训练模型
    train_losses, test_losses, learning_rates = train_model(
        model, train_loader, config, test_loader
    )

    # 绘制损失曲线
    plot_losses(train_losses, test_losses, config)

    # 绘制学习率变化曲线
    plot_learning_rate(learning_rates, config)

    # 评估模型并绘制对比图
    plot_comparison(model, X_test, y_test, config)

    # 绘制Q值对比图
    plot_q_comparison(model, config)

    # 最终测试损失
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    with torch.no_grad():
        test_loss = criterion(model(X_test.to(device)), y_test.to(device)).item()
    print(f"\n最终测试损失: {test_loss:.6f}")


if __name__ == "__main__":
    main()