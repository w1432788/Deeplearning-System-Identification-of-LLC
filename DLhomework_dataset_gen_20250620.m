%% 数据集生成器：M_DC(f_n, Q) 二元函数拟合（仅输出添加噪声）
% 目标函数：M_DC(f_n, Q) ≈ 0.94049 / sqrt( (6/5 - 1/(5*f_n^2))^2 + Q^2*( -1/f_n + f_n )^2 )
% 输出格式：CSV文件（可直接用PyTorch的DataLoader加载）
% 文件格式：[f_n, Q, M_DC] 三列数据，每行一个样本

%% 用户可配置参数
f_n_min = 0.1;         % f_n最小值（大于0）
f_n_max = 2.0;         % f_n最大值
Q_min = 0.1;           % Q最小值（大于0）
Q_max = 10;            % Q最大值
num_samples = 100000;  % 总样本数量
filename = 'M_DC_dataset.csv'; % 输出文件名

% 输出噪声参数配置
add_output_noise = true;      % 是否在输出(M_DC)添加噪声
output_noise_type = 'gaussian'; % 噪声类型：'gaussian'（高斯）或'uniform'（均匀）
output_noise_scale = 0.1;    % 噪声强度（建议0-0.1）

%% 参数验证
assert(f_n_min > 0, 'f_n_min必须大于0');
assert(Q_min > 0, 'Q_min必须大于0');
assert(num_samples > 100, '样本数量至少为100');
assert(ismember(output_noise_type, {'gaussian', 'uniform'}), '噪声类型必须为''gaussian''或''uniform''');
assert(output_noise_scale >= 0 && output_noise_scale <= 0.5, '噪声强度应在0-0.5范围内');

%% 生成随机样本点
rng(42); % 固定随机种子确保可复现性

% 输入特征采样
f_n_values = f_n_min + (f_n_max - f_n_min) * rand(num_samples, 1);
Q_values = Q_min + (Q_max - Q_min) * rand(num_samples, 1);

%% 计算目标函数值 M_DC（无噪声）
M_DC_theory = zeros(num_samples, 1);
for i = 1:num_samples
    f_n = f_n_values(i);
    Q = Q_values(i);
    
    term1 = (6/5 - 1/(5*f_n^2));
    term2 = Q * (f_n - 1/f_n);
    denominator = sqrt(term1^2 + term2^2);
    
    % 替换条件运算符为if-else
    if denominator < eps
        M_DC_theory(i) = 0;
    else
        M_DC_theory(i) = 0.94049 / denominator;
    end
end

%% 添加输出噪声（核心修改部分）
if add_output_noise
    % 计算M_DC的统计量（用于噪声尺度）
    M_DC_mean = mean(M_DC_theory);
    M_DC_std = std(M_DC_theory);
    
    % 根据指定类型添加噪声
    if strcmp(output_noise_type, 'gaussian')
        % 高斯噪声：均值为0，标准差与M_DC尺度相关
        noise = output_noise_scale * M_DC_std * randn(num_samples, 1);
    else
        % 均匀噪声：在[-scale, scale]范围内
        noise = 2 * output_noise_scale * M_DC_std * (rand(num_samples, 1) - 0.5);
    end
    
    % 添加噪声并确保物理合理性（M_DC≥0）
    M_DC_values = max(0, M_DC_theory + noise);
else
    M_DC_values = M_DC_theory;
end

%% 数据标准化
f_n_mean = mean(f_n_values);
f_n_std = std(f_n_values);
Q_mean = mean(Q_values);
Q_std = std(Q_values);
M_DC_mean = mean(M_DC_values);
M_DC_std = std(M_DC_values);

f_n_scaled = (f_n_values - f_n_mean) / f_n_std;
Q_scaled = (Q_values - Q_mean) / Q_std;
M_DC_scaled = (M_DC_values - M_DC_mean) / M_DC_std;

%% 保存数据集
data_matrix = [f_n_values, Q_values, M_DC_values]; % 含噪声的原始数据
scaled_data = [f_n_scaled, Q_scaled, M_DC_scaled]; % 标准化数据

% 保存CSV文件
header = {'f_n', 'Q', 'M_DC'};
csvwrite_with_headers(filename, data_matrix, header);

% 保存标准化参数
save('normalization_params_with_noise.mat', 'f_n_mean', 'f_n_std', 'Q_mean', 'Q_std', 'M_DC_mean', 'M_DC_std');

%% 可视化对比（噪声前后对比）
figure('Position', [100, 100, 1200, 500]);

% 左图：含噪声的M_DC
subplot(1,2,1);
scatter3(f_n_values(1:num_samples), Q_values(1:num_samples), M_DC_values(1:num_samples), 10, M_DC_values(1:num_samples), 'filled');
xlabel('f_n'); ylabel('Q'); zlabel('M_{DC}');
title(sprintf('含噪声的M_DC (scale=%.2f)', output_noise_scale));
colorbar;
grid on;

% 右图：理论M_DC（无噪声）
subplot(1,2,2);
scatter3(f_n_values(1:num_samples), Q_values(1:num_samples), M_DC_theory(1:num_samples), 10, M_DC_theory(1:num_samples), 'filled');
xlabel('f_n'); ylabel('Q'); zlabel('M_{DC}');
title('理论M_DC（无噪声）');
colorbar;
grid on;

%% 辅助函数
function csvwrite_with_headers(filename, data, header)
    fid = fopen(filename, 'w');
    fprintf(fid, '%s,%s,%s\n', header{1}, header{2}, header{3});
    fclose(fid);
    dlmwrite(filename, data, '-append', 'delimiter', ',');
end