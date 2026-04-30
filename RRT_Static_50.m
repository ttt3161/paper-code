%% 三维空间RRT批量性能分析脚本
%===================================================================

function Batch_Analysis_RRT_3D()

    clear;
    clc;
    close all;

    %% 1. 实验设置
    num_runs = 50; % <<< 设置您想要自动运行的总次数 >>>

    % 定义固定的环境和参数
    space.x_min = 0; space.x_max = 1000;
    space.y_min = 0; space.y_max = 1000;
    space.z_min = 0; space.z_max = 1000;
    start_point = [0, 0, 0];
    goal_point = [1000, 1000, 1000];
    obstacles.cubes = createCubeObject_local();
    obstacles.cylinders = createCylinderObject_local();
    obstacles.spheres = createSphereObject_local();

    params.step_size = 10;
    params.max_iter = 3000;
    params.goal_threshold = 10;
    params.goal_bias = 0.02;

    %% 2. 批量运行并收集数据
    results = struct('success', [], 'planning_time', [], 'iterations', [], ...
                     'total_nodes', [], 'node_utilization_rate', [], ...
                     'path_length', []);
    
    fprintf('开始进行 %d 次 三维RRT 批量实验...\n', num_runs);
    fprintf('=========================================\n');
    
    for i = 1:num_runs
        fprintf('--- 正在运行第 %d / %d 次实验 ---\n', i, num_runs);
        
        [res] = run_single_rrt_3d_local(space, start_point, goal_point, params, obstacles);
        
        results(i).success = res.success;
        results(i).planning_time = res.planning_time;
        results(i).iterations = res.iterations;
        results(i).total_nodes = res.total_nodes;
        
        if res.success
            results(i).node_utilization_rate = res.node_utilization_rate;
            results(i).path_length = res.path_length;
            
            fprintf('成功！迭代: %d, 时间: %.4fs, 节点数: %d, 利用率: %.2f%%, 路径长度: %.2f\n', ...
                    res.iterations, res.planning_time, res.total_nodes, res.node_utilization_rate, res.path_length);
        else
            fprintf('失败。迭代: %d, 时间: %.4f s, 节点数: %d\n', res.iterations, res.planning_time, res.total_nodes);
        end
    end
    
    fprintf('=========================================\n');
    fprintf('所有实验完成！\n\n');

    %% 3. 统计和分析结果
    successful_runs = results([results.success] == true);
    num_success = length(successful_runs);
    
    if num_success > 0
        success_rate = (num_success / num_runs) * 100;
        avg_time = mean([successful_runs.planning_time]);
        avg_iterations = mean([successful_runs.iterations]);
        avg_total_nodes = mean([successful_runs.total_nodes]);
        avg_path_length = mean([successful_runs.path_length]);
        avg_node_utilization_rate = mean([successful_runs.node_utilization_rate]);
        
        fprintf('--- 最终统计结果 (基于 %d 次成功运行) ---\n', num_success);
        fprintf('成功率:                 %.1f %%\n', success_rate);
        fprintf('平均迭代次数:           %.1f\n', avg_iterations);
        fprintf('平均规划时间:           %.4f 秒\n', avg_time);
        fprintf('平均RRT树节点数:        %.1f\n', avg_total_nodes);
        fprintf('平均节点利用率:         %.2f %%\n', avg_node_utilization_rate);
        fprintf('平均路径长度:           %.2f\n', avg_path_length);
    else
        fprintf('--- 最终统计结果 ---\n');
        fprintf('所有 %d 次实验均失败。\n', num_runs);
    end
end

%% ========================== 核心RRT规划工作函数 ==========================
function [result] = run_single_rrt_3d_local(space, start_point, goal_point, params, obstacles)
    tic;
    result = struct('success', false, 'planning_time', 0, 'iterations', 0, ...
                    'total_nodes', 0, 'path_length', inf, 'node_utilization_rate', 0);
    
    T.nodes(1, :) = start_point;
    T.parent(1) = 0;
    path_found = false;
    
    iter = 0;
    for iter_count = 1:params.max_iter
        rand_point = sample_point_3d_biased(space, goal_point, params.goal_bias);
        [near_node, near_idx] = find_nearest_node(rand_point, T);
        new_node = expand_node(near_node, rand_point, params.step_size);
        if is_collision_3d(near_node, new_node, obstacles), continue; end
        T.nodes(end+1, :) = new_node;
        T.parent(end+1) = near_idx;
        if pdist([new_node; goal_point], 'euclidean') < params.goal_threshold
            path_found = true;
            iter = iter_count;
            break;
        end
    end
    
    result.planning_time = toc;
    result.total_nodes = size(T.nodes, 1);
    result.iterations = iter_count;

    if path_found
        result.success = true;
        final_path = reconstruct_path(T, goal_point);
        result.path_length = calculate_path_length(final_path);
        result.node_utilization_rate = (size(final_path, 1) / result.total_nodes) * 100;
    end
end

%% ========================== 辅助与核心算法局部函数 ==========================
function len = calculate_path_length(path)
    len = 0; if isempty(path) || size(path, 1) < 2, return; end
    for i = 1:(size(path,1)-1), len = len + norm(path(i+1,:) - path(i,:)); end
end
function rand_p = sample_point_3d_biased(space, goal_p, goal_bias)
    if rand < goal_bias, rand_p = goal_p;
    else, rand_x = rand * (space.x_max - space.x_min) + space.x_min; rand_y = rand * (space.y_max - space.y_min) + space.y_min; rand_z = rand * (space.z_max - space.z_min) + space.z_min; rand_p = [rand_x, rand_y, rand_z]; end
end
function collision = is_collision_3d(start_n, end_n, obstacles)
    collision = false; num_checks = 15; line_points = linspace(0, 1, num_checks);
    for i = 2:num_checks
        point = start_n + line_points(i) * (end_n - start_n);
        if obstacles.spheres.exist, for k = 1:length(obstacles.spheres.centerX), center = [obstacles.spheres.centerX(k), obstacles.spheres.centerY(k), obstacles.spheres.centerZ(k)]; radius = obstacles.spheres.radius(k); if norm(point - center) <= radius, collision = true; return; end, end, end
        if obstacles.cubes.exist, for k = 1:length(obstacles.cubes.axisX), x_min = obstacles.cubes.axisX(k); x_max = x_min + obstacles.cubes.length(k); y_min = obstacles.cubes.axisY(k); y_max = y_min + obstacles.cubes.width(k); z_min = obstacles.cubes.axisZ(k); z_max = z_min + obstacles.cubes.height(k); if (point(1) >= x_min && point(1) <= x_max && point(2) >= y_min && point(2) <= y_max && point(3) >= z_min && point(3) <= z_max), collision = true; return; end, end, end
        if obstacles.cylinders.exist, for k = 1:length(obstacles.cylinders.X), center_xy = [obstacles.cylinders.X(k), obstacles.cylinders.Y(k)]; radius = obstacles.cylinders.radius(k); z_min = obstacles.cylinders.Z(k); z_max = z_min + obstacles.cylinders.height(k); if norm(point(1:2) - center_xy) <= radius && point(3) >= z_min && point(3) <= z_max, collision = true; return; end, end, end
    end
end
function [near_node, near_idx] = find_nearest_node(rand_p, T)
    distances = pdist2(rand_p, T.nodes); [~, near_idx] = min(distances); near_node = T.nodes(near_idx, :);
end
function new_node = expand_node(near_n, rand_p, step_size)
    direction_vec = rand_p - near_n; dist = norm(direction_vec); if dist < step_size, new_node = rand_p; else, unit_vec = direction_vec / dist; new_node = near_n + step_size * unit_vec; end
end
function path = reconstruct_path(T, goal_p)
    path = goal_p; last_node_dist = pdist2(goal_p, T.nodes); [~, current_idx] = min(last_node_dist);
    while current_idx ~= 0, current_node = T.nodes(current_idx, :); path = [current_node; path]; current_idx = T.parent(current_idx); end
end
function cubeInfo = createCubeObject_local()
    cubeInfo.axisX = [600 300 200]; cubeInfo.axisY = [300 400 800]; cubeInfo.axisZ = [300 100 400];
    cubeInfo.length = [200 200 100]; cubeInfo.width = [200 100 150]; cubeInfo.height = [200 200 100];
    cubeInfo.exist = 1;
end
function cylinderInfo = createCylinderObject_local()
    cylinderInfo.X = [600 300 500]; cylinderInfo.Y = [600 300 100]; cylinderInfo.Z = [700 100 200];
    cylinderInfo.radius = [50 20 50]; cylinderInfo.height = [200 100 250];
    cylinderInfo.exist = 1;
end
function sphereInfo = createSphereObject_local()
    sphereInfo.centerX = [600 800 400]; sphereInfo.centerY = [600 800 700]; sphereInfo.centerZ = [600 800 400];
    sphereInfo.radius = [90 80 70];
    sphereInfo.exist = 1;
end