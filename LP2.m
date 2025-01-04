data = load('cc200_fMRI.mat');
keys = fieldnames(data);
gamma = 0.1;
results = struct(); % 初始化一个结构体来存储结果
results_threshold = struct();
%for i = 1:length(keys)
for i = 1:length(keys)
    disp(keys{i});
    subject_data = data.(keys{i});
    subject_data = subject_data'; % 转置数据
    %disp(size(subject_data)); % 显示数据大小
    % 计算超边
    [hyperedges5, hyperedges10,hyperedges15,hyperedges20,hyperedges30,hyperedges50,hyperedgess1,hyperedgess2,hyperedgess3,hyperedgess4,hyperedgess5,hyperedgess6] = estimate_H(subject_data, gamma);
    results.(keys{i}) = struct('hyperedges5', hyperedges5, 'hyperedges10', hyperedges10,'hyperedges15', hyperedges15,'hyperedges20', hyperedges20,'hyperedges30', hyperedges30,'hyperedges50', hyperedges50); % 将结果存储在结构体中
    results_threshold.(keys{i}) = struct('hyperedgess1', hyperedgess1,'hyperedgess2', hyperedgess2,'hyperedgess3', hyperedgess3,'hyperedgess4', hyperedgess4,'hyperedgess5', hyperedgess5,'hyperedgess6', hyperedgess6);
    % 使用 sprintf 创建包含 K 和 threshold 的文件名
end        
filename = sprintf('cc200_topK.mat');
% 使用创建的文件名保存 results 变量
save(filename, 'results');
filename1 = sprintf('cc200_threshold.mat');
% 使用创建的文件名保存 results 变量
save(filename1, 'results_threshold');
% 定义估计关联矩阵H的函数
function [hyperedges5, hyperedges10,hyperedges15,hyperedges20,hyperedges30,hyperedges50,hyperedgess1,hyperedgess2,hyperedgess3,hyperedgess4,hyperedgess5,hyperedgess6] = estimate_H(R, gamma)
    N = size(R, 1); % ROIs的数量
    hyperedges5 = zeros(N, N);
    hyperedges10 = zeros(N, N);
    hyperedges15 = zeros(N, N);
    hyperedges20 = zeros(N, N);
    hyperedges30 = zeros(N, N);
    hyperedges50 = zeros(N, N);

    hyperedgess1 = zeros(N, N);
    hyperedgess2 = zeros(N, N);
    hyperedgess3 = zeros(N, N);
    hyperedgess4 = zeros(N, N);
    hyperedgess5 = zeros(N, N);
    hyperedgess6 = zeros(N, N);

    for i = 1:N
        R_i = R([1:i-1, i+1:end], :); % 删除第i个脑区的时间序列数据
        R_i_target = R(i, :); % 第i个脑区的时间序列数据作为目标
        %disp(R_i_target);
        disp(i);
        % 初始猜测e_i为全零向量
        e_i_init = rand(N-1, 1);
        %e_i_init = zeros(N-1, 1);
        % 优化问题求解e_i
        options = optimoptions('fmincon', 'Display', 'off');
         % 修改优化算法，例如设置为 'interior-point'（内点法）
        options.Algorithm = 'sqp';
        % 你也可以设置其他选项，例如最大迭代次数
        options.MaxIterations = 5;
        e_i = fmincon(@(e_i) objective(e_i, R_i, R_i_target, gamma), e_i_init, [], [], [], [], zeros(N-1, 1), [], [], options);
        % 假设 e_i 是一个向量，i 是要插入的位置
        % 将数值向量转换为单元格数组
        e_i_cell = num2cell(e_i);
        % 在第i个位置插入值10
        e_i_cell{i} = 10;
        % 将单元格数组转换回数值向量
        e_i = cell2mat(e_i_cell);
        % 根据e_i构建超边
        e_i_init = abs(e_i); % 插入第i个元素
        [~, top_K_indices1] = sort(e_i_init, 'descend');
        %disp(e_i_init);
        top_K_indices5 = top_K_indices1(1:6); % 获取最大的K个索引
        hyperedges5(top_K_indices5, i) = 1;
        
        top_K_indices10 = top_K_indices1(1:11); % 获取最大的K个索引
        hyperedges10(top_K_indices10, i) = 1;

        top_K_indices15 = top_K_indices1(1:16); % 获取最大的K个索引
        hyperedges15(top_K_indices15, i) = 1;

        top_K_indices20 = top_K_indices1(1:21); % 获取最大的K个索引
        hyperedges20(top_K_indices20, i) = 1;

        top_K_indices30 = top_K_indices1(1:31); % 获取最大的K个索引
        hyperedges30(top_K_indices30, i) = 1;

        top_K_indices50 = top_K_indices1(1:51); % 获取最大的K个索引
        hyperedges50(top_K_indices50, i) = 1;




        top_K_indicess1 = e_i_init >= 1; % 获取所有大于或等于阈值的索引
        hyperedgess1(top_K_indicess1, i) = 1;


        top_K_indicess2 = e_i_init >= 0.5; % 获取所有大于或等于阈值的索引
        hyperedgess2(top_K_indicess2, i) = 1;

        top_K_indicess3 = e_i_init >= 0.1; % 获取所有大于或等于阈值的索引
        hyperedgess3(top_K_indicess3, i) = 1;

        top_K_indicess4 = e_i_init >= 0.05; % 获取所有大于或等于阈值的索引
        hyperedgess4(top_K_indicess4, i) = 1;

        top_K_indicess5 = e_i_init >= 0.01; % 获取所有大于或等于阈值的索引
        hyperedgess5(top_K_indicess5, i) = 1;

        top_K_indicess6 = e_i_init >= 0.001; % 获取所有大于或等于阈值的索引
        hyperedgess6(top_K_indicess6, i) = 1;


    end
end

% 定义目标函数
function J = objective(e_i, R_i, R_i_target, gamma)
    R_i = R_i';
    e_i = reshape(e_i, [length(e_i), 1]);
    term1 = 0.5 * norm(R_i_target - R_i * e_i, 2)^2;
    term2 = gamma * norm(e_i, 1);
    J = double(term1 + term2); % 确保返回值是double类型
end