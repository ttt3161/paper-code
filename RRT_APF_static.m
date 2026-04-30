function Robust_APF_RRT_3D()

    clear;
    clc;
    close all;

    space.x_min = 0; space.x_max = 1000;
    space.y_min = 0; space.y_max = 1000;
    space.z_min = 0; space.z_max = 1000;
    
    start_point = [0, 0, 0];
    goal_point = [1000, 1000, 1000];
    
    obstacles.cubes = createCubeObject();
    obstacles.cylinders = createCylinderObject();
    obstacles.spheres = createSphereObject();
    
    params.step_size = 10;      
    params.max_iter = 3000;    
    params.goal_threshold = 10; 
    params.goal_bias = 0.02; 

    apf_params.w_att = 1.5;  
    apf_params.w_rep = 1;  
    apf_params.w_rand = 0.5;  
    apf_params.Kr = 1;     
    apf_params.rho0 = 100;    

    %% 
    fig = figure; hold on;
    axis([space.x_min, space.x_max, space.y_min, space.y_max, space.z_min, space.z_max]);
    axis equal; view(3); grid on;
    ax = gca; ax.FontSize = 16;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    
    scatter3(start_point(1), start_point(2), start_point(3), 100, 'go', 'MarkerFaceColor', 'g');
    scatter3(goal_point(1), goal_point(2), goal_point(3), 100, 'ro', 'MarkerFaceColor', 'r');

    draw_obstacles(obstacles);

    %% 
    T.nodes(1, :) = start_point;
    T.parent(1) = 0;

    fprintf('3D RRT planning started (Robust APF guided)...\n');
    path_found = false;
    
    planning_timer = tic; % <<< Start the timer

    for iter = 1:params.max_iter
        rand_point = sample_point_3d_biased(space, goal_point, params.goal_bias);
        
        [near_node, near_idx] = find_nearest_node(rand_point, T);

        new_node = expand_node_apf_guided_3d(near_node, rand_point, goal_point, obstacles, params, apf_params);
        
        if is_collision_3d(near_node, new_node, obstacles), continue; end

        T.nodes(end+1, :) = new_node;
        T.parent(end+1) = near_idx;
        
        plot3([near_node(1), new_node(1)], [near_node(2), new_node(2)], [near_node(3), new_node(3)], 'b-', 'LineWidth', 1);
        drawnow limitrate;
        
        if pdist([new_node; goal_point], 'euclidean') < params.goal_threshold
            path_found = true;
            break;
        end
    end

    planning_time = toc(planning_timer); % <<< Stop the timer

    %% 
    if path_found
        final_path = reconstruct_path(T, goal_point);
        
        % <<< Performance Metrics Calculation
        total_nodes = size(T.nodes, 1);
        path_length = 0;
        for k = 1:size(final_path, 1) - 1
            path_length = path_length + norm(final_path(k+1,:) - final_path(k,:));
        end
        utilization_rate = (size(final_path, 1) / total_nodes) * 100;
        
        fprintf('\n==================================\n');
        fprintf('        Path planning successful!\n');
        fprintf('==================================\n');
        fprintf('Number of iterations used:       %d\n', iter);
        fprintf('Time required for planning:      %.4f s\n', planning_time);
        fprintf('Final path length:               %.2f\n', path_length);
        fprintf('Total number of nodes generated: %d\n', total_nodes);
        fprintf('Node utilization:                %.2f %%\n', utilization_rate);
        fprintf('==================================\n\n');
        
        for k = 1:size(final_path, 1) - 1
            line([final_path(k,1), final_path(k+1,1)], ...
                 [final_path(k,2), final_path(k+1,2)], ...
                 [final_path(k,3), final_path(k+1,3)], ...
                 'LineWidth', 2.5, 'Color', 'red');
        end
    else
        fprintf('\nPlanning failed. Maximum iterations reached (%d iterations).\n', iter);
        fprintf('Total time consumed: %.4f s\n', planning_time);
        fprintf('Total nodes generated: %d\n', size(T.nodes, 1));
    end
    
    hold off;
end

function rand_p = sample_point_3d_biased(space, goal_p, goal_bias)
    if rand < goal_bias
        rand_p = goal_p;
    else
        rand_x = rand * (space.x_max - space.x_min) + space.x_min;
        rand_y = rand * (space.y_max - space.y_min) + space.y_min;
        rand_z = rand * (space.z_max - space.z_min) + space.z_min;
        rand_p = [rand_x, rand_y, rand_z];
    end
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
        unit_vec = F_rand;
        if norm(unit_vec) > 0, new_node = near_n + params.step_size * (unit_vec/norm(unit_vec)); else, new_node = near_n; end
    end
end

function F_rep_total = calculate_repulsive_force_3d(current_pos, obstacles, params)
    F_rep_total = [0, 0, 0];
    if obstacles.spheres.exist
        for i = 1:length(obstacles.spheres.centerX), center = [obstacles.spheres.centerX(i), obstacles.spheres.centerY(i), obstacles.spheres.centerZ(i)]; radius = obstacles.spheres.radius(i); dist_to_edge = norm(current_pos - center) - radius; if dist_to_edge < params.rho0 && dist_to_edge > 1e-6, grad_dir = (current_pos - center) / norm(current_pos - center); magnitude = params.Kr * (1/dist_to_edge - 1/params.rho0); F_rep_total = F_rep_total + magnitude * grad_dir; end, end
    end
    if obstacles.cubes.exist
        for i = 1:length(obstacles.cubes.axisX), x_min = obstacles.cubes.axisX(i); x_max = x_min + obstacles.cubes.length(i); y_min = obstacles.cubes.axisY(i); y_max = y_min + obstacles.cubes.width(i); z_min = obstacles.cubes.axisZ(i); z_max = z_min + obstacles.cubes.height(i); closest_x = max(x_min, min(current_pos(1), x_max)); closest_y = max(y_min, min(current_pos(2), y_max)); closest_z = max(z_min, min(current_pos(3), z_max)); closest_point = [closest_x, closest_y, closest_z]; dist_to_edge = norm(current_pos - closest_point); if dist_to_edge < params.rho0 && dist_to_edge > 1e-6, grad_dir = (current_pos - closest_point) / dist_to_edge; magnitude = params.Kr * (1/dist_to_edge - 1/params.rho0); F_rep_total = F_rep_total + magnitude * grad_dir; end, end
    end
    if obstacles.cylinders.exist
        for i = 1:length(obstacles.cylinders.X), center_xy = [obstacles.cylinders.X(i), obstacles.cylinders.Y(i)]; radius = obstacles.cylinders.radius(i); z_min = obstacles.cylinders.Z(i); z_max = z_min + obstacles.cylinders.height(i); dist_to_axis_xy = norm(current_pos(1:2) - center_xy); if current_pos(3) > z_max, closest_point_on_axis = [center_xy, z_max]; elseif current_pos(3) < z_min, closest_point_on_axis = [center_xy, z_min]; else, closest_point_on_axis = [center_xy, current_pos(3)]; end; if dist_to_axis_xy <= radius, closest_point_on_surface = closest_point_on_axis; else, vec_from_axis = (current_pos(1:2) - center_xy) / dist_to_axis_xy; closest_point_on_surface = closest_point_on_axis + [radius * vec_from_axis, 0]; end; dist_to_edge = norm(current_pos - closest_point_on_surface); if dist_to_edge < params.rho0 && dist_to_edge > 1e-6, grad_dir = (current_pos - closest_point_on_surface) / dist_to_edge; magnitude = params.Kr * (1/dist_to_edge - 1/params.rho0); F_rep_total = F_rep_total + magnitude * grad_dir; end, end
    end
end

function F_att = calculate_attractive_force_global(current_pos, goal_pos)
    F_att = goal_pos - current_pos;
end
function [near_node, near_idx] = find_nearest_node(rand_p, T)
    distances = pdist2(rand_p, T.nodes); [~, near_idx] = min(distances); near_node = T.nodes(near_idx, :);
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
function draw_obstacles(obstacles)
    pellucidity = 0.3;
    if obstacles.cubes.exist, for k = 1:length(obstacles.cubes.axisX), origin = [obstacles.cubes.axisX(k), obstacles.cubes.axisY(k), obstacles.cubes.axisZ(k)]; edges = [obstacles.cubes.length(k), obstacles.cubes.width(k), obstacles.cubes.height(k)]; plotcube(edges, origin, pellucidity, [1 1 0]); end, end
    if obstacles.cylinders.exist, for k = 1:length(obstacles.cylinders.X), center = [obstacles.cylinders.X(k), obstacles.cylinders.Y(k), obstacles.cylinders.Z(k)]; radius = obstacles.cylinders.radius(k); height = obstacles.cylinders.height(k); [x, y, z] = cylinder(radius, 30); z = z * height + center(3); surf(x + center(1), y + center(2), z, 'FaceColor', [0 1 0], 'EdgeColor', 'none', 'FaceAlpha', pellucidity); fill3(x(1,:) + center(1), y(1,:) + center(2), z(1,:), [0 1 0], 'FaceAlpha', pellucidity, 'EdgeColor', 'none'); fill3(x(2,:) + center(1), y(2,:) + center(2), z(2,:), [0 1 0], 'FaceAlpha', pellucidity, 'EdgeColor', 'none'); end, end
    if obstacles.spheres.exist, for k = 1:length(obstacles.spheres.centerX), center = [obstacles.spheres.centerX(k), obstacles.spheres.centerY(k), obstacles.spheres.centerZ(k)]; radius = obstacles.spheres.radius(k); [x, y, z] = sphere(50); surf(x*radius+center(1), y*radius+center(2), z*radius+center(3), 'FaceColor', [0 0 1], 'EdgeColor', 'none', 'FaceAlpha', pellucidity); end, end
end
function path = reconstruct_path(T, goal_p)
    path = goal_p; last_node_dist = pdist2(goal_p, T.nodes); [~, current_idx] = min(last_node_dist);
    while current_idx ~= 0, current_node = T.nodes(current_idx, :); path = [current_node; path]; current_idx = T.parent(current_idx); end
end
function cubeInfo = createCubeObject()
    cubeInfo.axisX = [600 300 200]; 
    cubeInfo.axisY = [300 400 800];
    cubeInfo.axisZ = [300 100 400]; 
    cubeInfo.length = [200 200 100]; 
    cubeInfo.width = [200 100 150]; 
    cubeInfo.height = [200 200 100]; 
    cubeInfo.exist = 1;
end
function cylinderInfo = createCylinderObject()
    cylinderInfo.X = [600 300 500]; 
    cylinderInfo.Y = [600 300 100]; 
    cylinderInfo.Z = [700 100 200]; 
    cylinderInfo.radius = [50 20 50]; 
    cylinderInfo.height = [200 100 250]; 
    cylinderInfo.exist = 1;
end
function sphereInfo = createSphereObject()
    sphereInfo.centerX = [600 800 400]; 
    sphereInfo.centerY = [600 800 700]; 
    sphereInfo.centerZ = [600 800 400]; 
    sphereInfo.radius = [90 80 70]; 
    sphereInfo.exist = 1;
end
function plotcube(varargin)
    inArgs = {[10 56 100],[10 10 10],.7,[1 0 0]}; inArgs(1:nargin) = varargin; [edges,origin,alpha,clr] = deal(inArgs{:});
    XYZ = {[0 0 0 0],[0 0 1 1],[0 1 1 0]; [1 1 1 1],[0 0 1 1],[0 1 1 0]; [0 1 1 0],[0 0 0 0],[0 0 1 1]; [0 1 1 0],[1 1 1 1],[0 0 1 1]; [0 1 1 0],[0 0 1 1],[0 0 0 0]; [0 1 1 0],[0 0 1 1],[1 1 1 1]};
    XYZ = mat2cell(cellfun( @(x,y,z) x*y+z, XYZ, repmat(mat2cell(edges,1,[1 1 1]),6,1), repmat(mat2cell(origin,1,[1 1 1]),6,1), 'UniformOutput',false), 6,[1 1 1]);
    cellfun(@patch,XYZ{1},XYZ{2},XYZ{3}, repmat({clr},6,1), repmat({'FaceAlpha'},6,1), repmat({alpha},6,1));
end