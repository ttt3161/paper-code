function Batch_Analysis_APF_RRT_3D_Fixed()

    clear;
    clc;
    close all;

    num_runs = 50;

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

    apf_params.w_att = 1.5;
    apf_params.w_rep = 1;
    apf_params.w_rand = 0.5;
    apf_params.Kr = 1;
    apf_params.rho0 = 100;

    results = struct('success', [], 'planning_time', [], 'iterations', [], ...
                     'total_nodes', [], 'node_utilization_rate', [], ...
                     'path_length', []);
    
    fprintf('Starting %d batch experiments of APF-guided 3D RRT...\n', num_runs);
    fprintf('=========================================\n');
    
    for i = 1:num_runs
        fprintf('--- Running experiment %d / %d ---\n', i, num_runs);
        
        [res] = run_single_apf_rrt_3d_local(space, start_point, goal_point, params, apf_params, obstacles);
        
        results(i).success = res.success;
        results(i).planning_time = res.planning_time;
        results(i).iterations = res.iterations;
        results(i).total_nodes = res.total_nodes;
        
        if res.success
            results(i).node_utilization_rate = res.node_utilization_rate;
            results(i).path_length = res.path_length;
            
            fprintf('Success! Iterations: %d, Time: %.4fs, Nodes: %d, Utilization: %.2f%%, Path Length: %.2f\n', ...
                    res.iterations, res.planning_time, res.total_nodes, res.node_utilization_rate, res.path_length);
        else
            fprintf('Failed. Iterations: %d, Time: %.4f s, Nodes: %d\n', res.iterations, res.planning_time, res.total_nodes);
        end
    end
    
    fprintf('=========================================\n');
    fprintf('All experiments completed!\n\n');

    successful_runs = results([results.success] == true);
    num_success = length(successful_runs);
    
    if num_success > 0
        success_rate = (num_success / num_runs) * 100;
        avg_time = mean([successful_runs.planning_time]);
        avg_iterations = mean([successful_runs.iterations]);
        avg_total_nodes = mean([successful_runs.total_nodes]);
        avg_path_length = mean([successful_runs.path_length]);
        avg_node_utilization_rate = mean([successful_runs.node_utilization_rate]);
        
        fprintf('--- Final Statistical Results (based on %d successful runs) ---\n', num_success);
        fprintf('Success Rate:                    %.1f %%\n', success_rate);
        fprintf('Average Iterations:              %.1f\n', avg_iterations);
        fprintf('Average Planning Time:           %.4f s\n', avg_time);
        fprintf('Average RRT Tree Nodes:          %.1f\n', avg_total_nodes);
        fprintf('Average Node Utilization Rate:   %.2f %%\n', avg_node_utilization_rate);
        fprintf('Average Path Length:             %.2f\n', avg_path_length);
    else
        fprintf('--- Final Statistical Results ---\n');
        fprintf('All %d experiments failed.\n', num_runs);
    end
end

function [result] = run_single_apf_rrt_3d_local(space, start_point, goal_point, params, apf_params, obstacles)
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
        new_node = expand_node_apf_guided_3d(near_node, rand_point, goal_point, obstacles, params, apf_params);
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

function len = calculate_path_length(path)
    len = 0; if isempty(path) || size(path, 1) < 2, return; end
    for i = 1:(size(path,1)-1), len = len + norm(path(i+1,:) - path(i,:)); end
end
function rand_p = sample_point_3d_biased(space, goal_p, goal_bias)
    if rand < goal_bias, rand_p = goal_p;
    else, rand_x = rand * (space.x_max - space.x_min) + space.x_min; rand_y = rand * (space.y_max - space.y_min) + space.y_min; rand_z = rand * (space.z_max - space.z_min) + space.z_min; rand_p = [rand_x, rand_y, rand_z]; end
end
function new_node = expand_node_apf_guided_3d(near_n, rand_p, goal_p, obstacles, params, apf_params)
    F_att = calculate_attractive_force_global(near_n, goal_p);
    F_rep = calculate_repulsive_force_3d(near_n, obstacles, apf_params);
    F_rand = rand_p - near_n;
    if norm(F_att) > 0, F_att = F_att / norm(F_att); end
    if norm(F_rep) > 0, F_rep = F_rep / norm(F_rep); end
    if norm(F_rand) > 0, F_rand = F_rand / norm(F_rand); end
    final_direction = apf_params.w_att * F_att + apf_params.w_rep * F_rep + apf_params.w_rand * F_rand;
    if norm(final_direction) > 0
        unit_vec = final_direction / norm(final_direction);
        new_node = near_n + params.step_size * unit_vec;
    else
        unit_vec = F_rand; if norm(unit_vec) > 0, new_node = near_n + params.step_size * (unit_vec/norm(unit_vec)); else, new_node = near_n; end
    end
end
function F_att = calculate_attractive_force_global(current_pos, goal_pos)
    F_att = goal_pos - current_pos;
end
function F_rep_total = calculate_repulsive_force_3d(current_pos, obstacles, params)
    F_rep_total = [0, 0, 0];
    if obstacles.spheres.exist, for i = 1:length(obstacles.spheres.centerX), center = [obstacles.spheres.centerX(i), obstacles.spheres.centerY(i), obstacles.spheres.centerZ(i)]; radius = obstacles.spheres.radius(i); dist_to_edge = norm(current_pos - center) - radius; if dist_to_edge < params.rho0 && dist_to_edge > 1e-6, grad_dir = (current_pos - center) / norm(current_pos - center); magnitude = params.Kr * (1/dist_to_edge - 1/params.rho0); F_rep_total = F_rep_total + magnitude * grad_dir; end, end, end
    if obstacles.cubes.exist, for i = 1:length(obstacles.cubes.axisX), x_min = obstacles.cubes.axisX(i); x_max = x_min + obstacles.cubes.length(i); y_min = obstacles.cubes.axisY(i); y_max = y_min + obstacles.cubes.width(i); z_min = obstacles.cubes.axisZ(i); z_max = z_min + obstacles.cubes.height(i); closest_x = max(x_min, min(current_pos(1), x_max)); closest_y = max(y_min, min(current_pos(2), y_max)); closest_z = max(z_min, min(current_pos(3), z_max)); closest_point = [closest_x, closest_y, closest_z]; dist_to_edge = norm(current_pos - closest_point); if dist_to_edge < params.rho0 && dist_to_edge > 1e-6, grad_dir = (current_pos - closest_point) / dist_to_edge; magnitude = params.Kr * (1/dist_to_edge - 1/params.rho0); F_rep_total = F_rep_total + magnitude * grad_dir; end, end, end
    if obstacles.cylinders.exist, for i = 1:length(obstacles.cylinders.X), center_xy = [obstacles.cylinders.X(i), obstacles.cylinders.Y(i)]; radius = obstacles.cylinders.radius(i); z_min = obstacles.cylinders.Z(i); z_max = z_min + obstacles.cylinders.height(i); dist_to_axis_xy = norm(current_pos(1:2) - center_xy); if current_pos(3) > z_max, closest_point_on_axis = [center_xy, z_max]; elseif current_pos(3) < z_min, closest_point_on_axis = [center_xy, z_min]; else, closest_point_on_axis = [center_xy, current_pos(3)]; end; if dist_to_axis_xy <= radius, closest_point_on_surface = closest_point_on_axis; else, vec_from_axis = (current_pos(1:2) - center_xy) / dist_to_axis_xy; closest_point_on_surface = closest_point_on_axis + [radius * vec_from_axis, 0]; end; dist_to_edge = norm(current_pos - closest_point_on_surface); if dist_to_edge < params.rho0 && dist_to_edge > 1e-6, grad_dir = (current_pos - closest_point_on_surface) / dist_to_edge; magnitude = params.Kr * (1/dist_to_edge - 1/params.rho0); F_rep_total = F_rep_total + magnitude * grad_dir; end, end, end
end
function [near_node, near_idx] = find_nearest_node(rand_p, T)
    distances = pdist2(rand_p, T.nodes); [~, near_idx] = min(distances); near_node = T.nodes(near_idx, :);
end
function collision = is_collision_3d(start_n, end_n, obstacles)
    collision = false; num_checks = 15; line_points = linspace(0, 1, num_checks);
    for i = 2:num_checks
        point = start_n + line_points(i) * (end_n - start_n);
        if obstacles.spheres.exist, for k = 1:length(obstacles.spheres.centerX), center = [obstacles.spheres.centerX(k), obstacles.spheres.centerY(k), obstacles.spheres.centerZ(k)]; radius = obstacles.spheres.radius(k); if norm(point - center) <= radius, collision = true; return; end, end, end
        if obstacles.cubes.exist
            for k = 1:length(obstacles.cubes.axisX)
                x_min = obstacles.cubes.axisX(k);
                x_max = x_min + obstacles.cubes.length(k);

                y_min = obstacles.cubes.axisY(k);
                y_max = y_min + obstacles.cubes.width(k);
                z_min = obstacles.cubes.axisZ(k);
                z_max = z_min + obstacles.cubes.height(k);
                if (point(1) >= x_min && point(1) <= x_max && ...
                    point(2) >= y_min && point(2) <= y_max && ...
                    point(3) >= z_min && point(3) <= z_max)
                    collision = true; return;
                end
            end
        end
        if obstacles.cylinders.exist, for k = 1:length(obstacles.cylinders.X), center_xy = [obstacles.cylinders.X(k), obstacles.cylinders.Y(k)]; radius = obstacles.cylinders.radius(k); z_min = obstacles.cylinders.Z(k); z_max = z_min + obstacles.cylinders.height(k); if norm(point(1:2) - center_xy) <= radius && point(3) >= z_min && point(3) <= z_max, collision = true; return; end, end, end
    end
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