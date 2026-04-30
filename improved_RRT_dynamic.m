function RRT_Full_Pipeline_3D_Final()

    clear;
    clc;
    close all;

    %%
    space.x_min = 0; space.x_max = 1000;
    space.y_min = 0; space.y_max = 1000;
    space.z_min = 0; space.z_max = 1000;
    
    start_point = [0, 0, 0];
    goal_point = [1000, 1000, 1000];
    
    obstacles.cubes = createCubeObject();
    obstacles.cylinders = createCylinderObject();
    obstacles.spheres = createSphereObject();
    

    rrt_params.goal_bias = 0.1; rrt_params.max_iter = 3000; rrt_params.goal_threshold = 10;
    apf_guide_params.w_att = 1.5; apf_guide_params.w_rep = 1; apf_guide_params.w_h = 0.5;
    apf_guide_params.Kr = 1; apf_guide_params.rho0 = 100;
    step_params.epsilon_max = 25; step_params.epsilon_min = 10;
    step_params.D_safe = 150;
    step_params.d_total_initial = norm(start_point - goal_point);
    apf_optim_params.Ka = 0.5; apf_optim_params.Kr = 1; 
    apf_optim_params.rho0 = 150;
    apf_optim_params.step_size = 8; apf_optim_params.max_steps = 200; 
    apf_optim_params.goal_threshold = 10; 


    fig = figure; hold on;
    axis([space.x_min, space.x_max, space.y_min, space.y_max, space.z_min, space.z_max]);
    axis equal; view(3); grid on;
    
    ax = gca; ax.FontSize = 16;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    
    scatter3(start_point(1), start_point(2), start_point(3), 100, 'go', 'MarkerFaceColor', 'g');
    scatter3(goal_point(1), goal_point(2), goal_point(3), 100, 'ro', 'MarkerFaceColor', 'r');
    
    h_final_path = plot3(NaN,NaN,NaN,'r-', 'LineWidth', 2.5);

    planning_timer = tic; 




    [T, obstacles, traj, final_iter] = find_initial_path_rrt_3d(space, start_point, goal_point, rrt_params, step_params, apf_guide_params, obstacles);
    
    if isempty(T)
        fprintf('\nRRT failed to find the initial path (reached maximum %d iterations); the program terminated.\n', rrt_params.max_iter); 
        draw_obstacles(obstacles); 
        hold off; 
        return; 
    end
    
    original_path = reconstruct_path(T, goal_point);

    fprintf('Path pruning is in progress...\n');
    pruned_path = prune_path_3d(original_path, obstacles);
    
    fprintf('APF post-processing optimization is in progress...\n');
    apf_optimized_path = APF_Post_Optimizer_3d(pruned_path, obstacles, apf_optim_params);
    
    fprintf('B-spline smoothing is in progress...\n');
    final_smooth_path = B_spline_smoother_cscvn_3d(apf_optimized_path, obstacles);

    planning_time = toc(planning_timer);
   

    if ~isempty(final_smooth_path)

        set(h_final_path, 'XData', final_smooth_path(:,1), 'YData', final_smooth_path(:,2), 'ZData', final_smooth_path(:,3));

        total_nodes = size(T.nodes, 1);
        path_length = 0;
        for k = 1:size(final_smooth_path, 1) - 1
            path_length = path_length + norm(final_smooth_path(k+1,:) - final_smooth_path(k,:));
        end

        utilization_rate = (size(original_path, 1) / total_nodes) * 100;
        
        fprintf('\n==================================\n');
        fprintf('        Path planning successful!\n');
        fprintf('==================================\n');
        fprintf('Actual iterations used:          %d\n', final_iter); 
        fprintf('Planning and smoothing time:     %.4f s\n', planning_time);
        fprintf('Final smoothed path length:      %.2f\n', path_length);
        fprintf('Total number of nodes generated: %d\n', total_nodes);
        fprintf('Original path utilization rate:  %.2f %%\n', utilization_rate);
        fprintf('==================================\n\n');

        for k = 1:3
            if size(traj.spheres{k}, 1) > 1
                plot3(traj.spheres{k}(:,1), traj.spheres{k}(:,2), traj.spheres{k}(:,3), 'm-.', 'LineWidth', 1.5);
            end
            if size(traj.cubes{k}, 1) > 1
                plot3(traj.cubes{k}(:,1), traj.cubes{k}(:,2), traj.cubes{k}(:,3), 'm-.', 'LineWidth', 1.5);
            end
            if size(traj.cylinders{k}, 1) > 1
                plot3(traj.cylinders{k}(:,1), traj.cylinders{k}(:,2), traj.cylinders{k}(:,3), 'm-.', 'LineWidth', 1.5);
            end
        end
    else
        fprintf('B-spline smoothing failed or path collision occurred.\n');
        fprintf('Total time consumed: %.4f s\n', planning_time);
    end
    
    draw_obstacles(obstacles);
    hold off;
end

function [T, obstacles, traj, iter] = find_initial_path_rrt_3d(space, start_point, goal_point, rrt_params, step_params, apf_guide_params, obstacles)
    T.nodes(1, :) = start_point; T.parent(1) = 0;
    try, halton_stream = haltonset(3); halton_stream = scramble(halton_stream, 'RR2'); 
    catch, error('Please install the Statistics and Machine Learning Toolbox™.'); end
    fprintf('3D RRT planning started (fully dynamic environment and trajectory recording)...\n');
    
    traj.spheres = cell(1,3); 
    traj.cubes = cell(1,3); 
    traj.cylinders = cell(1,3);

    dyn_obs = obstacles;

    for iter = 1:rrt_params.max_iter
        

        if mod(iter, 500) == 0
            fprintf('  -> Exploring... Current iteration: %d\n', iter);
        end
        
        fixed_dist = 250; 
        N_steps = 80;     
        cycle = mod(iter, 2 * N_steps);
        if cycle <= N_steps
            ratio = cycle / N_steps; 
        else
            ratio = (2 * N_steps - cycle) / N_steps; 
        end
        
        dyn_obs.spheres.centerX(1) = obstacles.spheres.centerX(1) + ratio * fixed_dist;
        dyn_obs.spheres.centerY(1) = obstacles.spheres.centerY(1) + ratio * fixed_dist;
        dyn_obs.spheres.centerX(2) = obstacles.spheres.centerX(2) - ratio * fixed_dist;
        dyn_obs.spheres.centerZ(2) = obstacles.spheres.centerZ(2) + ratio * fixed_dist;
        dyn_obs.spheres.centerY(3) = obstacles.spheres.centerY(3) - ratio * fixed_dist;
        dyn_obs.spheres.centerZ(3) = obstacles.spheres.centerZ(3) - ratio * fixed_dist;

        dyn_obs.cubes.axisY(1) = obstacles.cubes.axisY(1) + ratio * fixed_dist;
        dyn_obs.cubes.axisX(2) = obstacles.cubes.axisX(2) - ratio * fixed_dist;
        dyn_obs.cubes.axisZ(3) = obstacles.cubes.axisZ(3) + ratio * fixed_dist;

        dyn_obs.cylinders.X(1) = obstacles.cylinders.X(1) + ratio * fixed_dist;
        dyn_obs.cylinders.Y(2) = obstacles.cylinders.Y(2) - ratio * fixed_dist;
        dyn_obs.cylinders.X(3) = obstacles.cylinders.X(3) - ratio * fixed_dist;

        if mod(iter, 5) == 0
            for k = 1:3
                traj.spheres{k} = [traj.spheres{k}; dyn_obs.spheres.centerX(k), dyn_obs.spheres.centerY(k), dyn_obs.spheres.centerZ(k)];
                cx = dyn_obs.cubes.axisX(k) + dyn_obs.cubes.length(k)/2;
                cy = dyn_obs.cubes.axisY(k) + dyn_obs.cubes.width(k)/2;
                cz = dyn_obs.cubes.axisZ(k) + dyn_obs.cubes.height(k)/2;
                traj.cubes{k} = [traj.cubes{k}; cx, cy, cz];
                cyx = dyn_obs.cylinders.X(k);
                cyy = dyn_obs.cylinders.Y(k);
                cyz = dyn_obs.cylinders.Z(k) + dyn_obs.cylinders.height(k)/2;
                traj.cylinders{k} = [traj.cylinders{k}; cyx, cyy, cyz];
            end
        end

        rand_point = sample_point_halton_3d(space, goal_point, rrt_params.goal_bias, halton_stream, iter);
        [near_node, near_idx] = find_nearest_node(rand_point, T);
        
        new_node = expand_node_apf_guided_adaptive_3d(near_node, rand_point, goal_point, dyn_obs, step_params, apf_guide_params);
        
        if is_collision_3d(near_node, new_node, dyn_obs), continue; end
        T.nodes(end+1, :) = new_node; T.parent(end+1) = near_idx;
        plot3([near_node(1), new_node(1)], [near_node(2), new_node(2)], [near_node(3), new_node(3)], 'b-', 'HandleVisibility', 'off');
        drawnow limitrate;
        if pdist([new_node; goal_point], 'euclidean') < rrt_params.goal_threshold
            return; 
        end
    end
    T = [];
end

function final_path = APF_Post_Optimizer_3d(guideline_path, obstacles, params)
    current_pos = guideline_path(1,:);
    final_path = current_pos;
    for i = 2:size(guideline_path, 1)
        sub_goal = guideline_path(i,:);
        for step = 1:params.max_steps
            if norm(current_pos - sub_goal) < params.goal_threshold, break; end
            F_att = params.Ka * (sub_goal - current_pos);
            F_rep = calculate_repulsive_force_optimizer_3d(current_pos, obstacles, params);
            F_total = F_att + F_rep;
            if norm(F_total) > 0, direction = F_total / norm(F_total); current_pos = current_pos + params.step_size * direction; end
            final_path = [final_path; current_pos];
        end
    end
end


function smooth_path = B_spline_smoother_cscvn_3d(control_path, obstacles)
    smooth_path = [];
    if size(control_path, 1) < 3
        smooth_path = control_path; 
        return; 
    end
    
    smoothed_control_points = smooth_control_points_3d(control_path, 5);

    cleaned_points = smoothed_control_points(1, :);
    for i = 2:size(smoothed_control_points, 1)
        if norm(smoothed_control_points(i,:) - cleaned_points(end,:)) > 0.1
            cleaned_points = [cleaned_points; smoothed_control_points(i,:)];
        end
    end
    
    if size(cleaned_points, 1) < 3
        smooth_path = cleaned_points; return;
    end

    points = cleaned_points';
    
    try
        spline_curve = cscvn(points); 
    catch
        fprintf('\n[Warning] B-spline underlying mathematical poles, degrading to output original path.\n');
        smooth_path = cleaned_points; 
        return; 
    end
    
    t_vector = linspace(spline_curve.breaks(1), spline_curve.breaks(end), 500);
    smooth_points = fnval(spline_curve, t_vector);
    

    smooth_path = smooth_points';
end

function pruned_path = prune_path_3d(original_path, obstacles)
    if size(original_path, 1) <= 2, pruned_path = original_path; return; end
    pruned_path = original_path(1,:);
    current_index = 1;
    while current_index < size(original_path, 1)
        anchor_node = original_path(current_index, :);
        best_lookahead_index = current_index + 1;
        for lookahead_index = size(original_path, 1):-1:(current_index + 1)
            if ~is_collision_3d(anchor_node, original_path(lookahead_index, :), obstacles)
                best_lookahead_index = lookahead_index; break;
            end
        end
        pruned_path = [pruned_path; original_path(best_lookahead_index, :)];
        current_index = best_lookahead_index;
    end
end


function new_node = expand_node_apf_guided_adaptive_3d(near_n, rand_p, goal_p, obstacles, step_params, apf_params)
  d_obs = find_nearest_obstacle_dist_3d(near_n, obstacles);
if d_obs <= step_params.D_safe
    epsilon_obs = step_params.epsilon_min + (d_obs / step_params.D_safe) * (step_params.epsilon_max - step_params.epsilon_min);
else
    epsilon_obs = step_params.epsilon_max;
end
    d_goal = norm(near_n - goal_p);
    epsilon_goal = step_params.epsilon_min + (step_params.epsilon_max - step_params.epsilon_min) * (d_goal / step_params.d_total_initial);
    current_step_size = max(step_params.epsilon_min, min([step_params.epsilon_max, epsilon_obs, epsilon_goal]));
    F_att = calculate_attractive_force_global(near_n, goal_p);
    F_rep = calculate_repulsive_force_expand_3d(near_n, obstacles, apf_params);
    F_rand = rand_p - near_n;
    if norm(F_att) > 0, F_att = F_att / norm(F_att); end
    if norm(F_rep) > 0, F_rep = F_rep / norm(F_rep); end
    if norm(F_rand) > 0, F_rand = F_rand / norm(F_rand); end
    final_direction = apf_params.w_att * F_att + apf_params.w_rep * F_rep + apf_params.w_h * F_rand;
    if norm(final_direction) > 0, unit_vec = final_direction / norm(final_direction); new_node = near_n + current_step_size * unit_vec; else, unit_vec = F_rand; if norm(unit_vec)>0, new_node = near_n + current_step_size * (unit_vec/norm(unit_vec)); else, new_node = near_n; end; end
end


function F_rep_total = calculate_repulsive_force_optimizer_3d(current_pos, obstacles, params)
    F_rep_total = [0, 0, 0];
    if obstacles.spheres.exist, for i=1:length(obstacles.spheres.centerX), center=[obstacles.spheres.centerX(i),obstacles.spheres.centerY(i),obstacles.spheres.centerZ(i)]; radius=obstacles.spheres.radius(i); dist_to_edge=norm(current_pos-center)-radius; if dist_to_edge<params.rho0&&dist_to_edge>1e-6, grad_dir=(current_pos-center)/norm(current_pos-center); magnitude=params.Kr*(1/dist_to_edge-1/params.rho0); F_rep_total=F_rep_total+magnitude*grad_dir; end, end, end
    if obstacles.cubes.exist, for i=1:length(obstacles.cubes.axisX), x_min=obstacles.cubes.axisX(i);x_max=x_min+obstacles.cubes.length(i);y_min=obstacles.cubes.axisY(i);y_max=y_min+obstacles.cubes.width(i);z_min=obstacles.cubes.axisZ(i);z_max=z_min+obstacles.cubes.height(i); closest_x=max(x_min,min(current_pos(1),x_max));closest_y=max(y_min,min(current_pos(2),y_max));closest_z=max(z_min,min(current_pos(3),z_max)); closest_point=[closest_x,closest_y,closest_z]; dist_to_edge=norm(current_pos-closest_point); if dist_to_edge<params.rho0&&dist_to_edge>1e-6, grad_dir=(current_pos-closest_point)/dist_to_edge; magnitude=params.Kr*(1/dist_to_edge-1/params.rho0); F_rep_total=F_rep_total+magnitude*grad_dir; end, end, end
end

function F_rep_total = calculate_repulsive_force_expand_3d(current_pos, obstacles, params)
    F_rep_total = [0, 0, 0];
    if obstacles.spheres.exist, for i=1:length(obstacles.spheres.centerX), center=[obstacles.spheres.centerX(i),obstacles.spheres.centerY(i),obstacles.spheres.centerZ(i)]; radius=obstacles.spheres.radius(i); dist_to_edge=norm(current_pos-center)-radius; if dist_to_edge<params.rho0&&dist_to_edge>1e-6, grad_dir=(current_pos-center)/norm(current_pos-center); magnitude=params.Kr*(1/dist_to_edge-1/params.rho0); F_rep_total=F_rep_total+magnitude*grad_dir; end, end, end
    if obstacles.cubes.exist, for i=1:length(obstacles.cubes.axisX), x_min=obstacles.cubes.axisX(i);x_max=x_min+obstacles.cubes.length(i);y_min=obstacles.cubes.axisY(i);y_max=y_min+obstacles.cubes.width(i);z_min=obstacles.cubes.axisZ(i);z_max=z_min+obstacles.cubes.height(i); closest_x=max(x_min,min(current_pos(1),x_max));closest_y=max(y_min,min(current_pos(2),y_max));closest_z=max(z_min,min(current_pos(3),z_max)); closest_point=[closest_x,closest_y,closest_z]; dist_to_edge=norm(current_pos-closest_point); if dist_to_edge<params.rho0&&dist_to_edge>1e-6, grad_dir=(current_pos-closest_point)/dist_to_edge; magnitude=params.Kr*(1/dist_to_edge-1/params.rho0); F_rep_total=F_rep_total+magnitude*grad_dir; end, end, end
end

function F_att = calculate_attractive_force_global(current_pos, goal_pos)
    F_att = goal_pos - current_pos;
end

function smoothed_path = smooth_control_points_3d(path, window_size)
    if mod(window_size, 2) == 0, error('Window size must be an odd number.'); end
    half_window = floor(window_size / 2); n_points = size(path, 1);
    smoothed_path = path;
    for i = 2:(n_points - 1), start_idx = max(1, i - half_window); end_idx = min(n_points, i + half_window); window_points = path(start_idx:end_idx, :); smoothed_path(i, :) = mean(window_points, 1); end
end

function rand_p = sample_point_halton_3d(space, goal_p, goal_bias, halton_stream, iter)
    if rand < goal_bias, rand_p = goal_p;
    else, p_unit = halton_stream(iter, :); rand_p(1) = p_unit(1) * (space.x_max - space.x_min) + space.x_min; rand_p(2) = p_unit(2) * (space.y_max - space.y_min) + space.y_min; rand_p(3) = p_unit(3) * (space.z_max - space.z_min) + space.z_min; end
end

function min_dist = find_nearest_obstacle_dist_3d(point, obstacles)
    min_dist = inf;
    if obstacles.spheres.exist, for i = 1:length(obstacles.spheres.centerX), center = [obstacles.spheres.centerX(i),obstacles.spheres.centerY(i),obstacles.spheres.centerZ(i)]; dist = norm(point - center) - obstacles.spheres.radius(i); if dist < min_dist, min_dist = dist; end, end, end
    if obstacles.cubes.exist, for i = 1:length(obstacles.cubes.axisX), rect_min = [obstacles.cubes.axisX(i), obstacles.cubes.axisY(i), obstacles.cubes.axisZ(i)]; rect_max = rect_min + [obstacles.cubes.length(i), obstacles.cubes.width(i), obstacles.cubes.height(i)]; d = max(rect_min - point, max(0, point - rect_max)); dist = norm(d); if dist < min_dist, min_dist = dist; end, end, end
end

function [near_node, near_idx] = find_nearest_node(rand_p, T)
    distances = pdist2(rand_p, T.nodes); [~, near_idx] = min(distances); near_node = T.nodes(near_idx, :);
end

function collision = is_collision_3d(start_n, end_n, obstacles)
    collision = false; segment_length = norm(end_n - start_n); resolution = 8;
    num_checks = ceil(segment_length / resolution); if num_checks < 2, num_checks = 2; end 
    line_points = linspace(0, 1, num_checks);
    for i = 1:num_checks
        point = start_n + line_points(i) * (end_n - start_n);
        if obstacles.spheres.exist, for k=1:length(obstacles.spheres.centerX), center=[obstacles.spheres.centerX(k),obstacles.spheres.centerY(k),obstacles.spheres.centerZ(k)]; if norm(point-center)<=obstacles.spheres.radius(k), collision=true; return; end, end, end
        if obstacles.cubes.exist, for k=1:length(obstacles.cubes.axisX), x_min=obstacles.cubes.axisX(k); x_max=x_min+obstacles.cubes.length(k); y_min=obstacles.cubes.axisY(k); y_max=y_min+obstacles.cubes.width(k); z_min=obstacles.cubes.axisZ(k); z_max=z_min+obstacles.cubes.height(k); if (point(1)>=x_min&&point(1)<=x_max&&point(2)>=y_min&&point(2)<=y_max&&point(3)>=z_min&&point(3)<=z_max), collision=true; return; end, end, end
        if obstacles.cylinders.exist, for k=1:length(obstacles.cylinders.X), center_xy=[obstacles.cylinders.X(k),obstacles.cylinders.Y(k)]; radius=obstacles.cylinders.radius(k); z_min=obstacles.cylinders.Z(k); z_max=z_min+obstacles.cylinders.height(k); if norm(point(1:2)-center_xy)<=radius&&point(3)>=z_min&&point(3)<=z_max, collision=true; return; end, end, end
    end
end

function path = reconstruct_path(T, goal_p)
    path = goal_p; last_node_dist = pdist2(goal_p, T.nodes); [~, current_idx] = min(last_node_dist);
    while current_idx ~= 0, current_node = T.nodes(current_idx, :); path = [current_node; path]; current_idx = T.parent(current_idx); end
end


function draw_obstacles(obstacles)
    pellucidity = 0.3;
    if obstacles.cubes.exist, for k=1:length(obstacles.cubes.axisX), origin=[obstacles.cubes.axisX(k),obstacles.cubes.axisY(k),obstacles.cubes.axisZ(k)]; edges=[obstacles.cubes.length(k),obstacles.cubes.width(k),obstacles.cubes.height(k)]; plotcube(edges, origin, pellucidity, [1 1 0]); end, end
    if obstacles.cylinders.exist, for k=1:length(obstacles.cylinders.X), center=[obstacles.cylinders.X(k),obstacles.cylinders.Y(k),obstacles.cylinders.Z(k)]; radius=obstacles.cylinders.radius(k); height=obstacles.cylinders.height(k); [x,y,z]=cylinder(radius,30); z=z*height+center(3); surf(x+center(1),y+center(2),z,'FaceColor',[0 1 0],'EdgeColor','none','FaceAlpha',pellucidity); fill3(x(1,:)+center(1),y(1,:)+center(2),z(1,:),[0 1 0],'FaceAlpha',pellucidity,'EdgeColor','none'); fill3(x(2,:)+center(1),y(2,:)+center(2),z(2,:),[0 1 0],'FaceAlpha',pellucidity,'EdgeColor','none'); end, end
    if obstacles.spheres.exist, for k=1:length(obstacles.spheres.centerX), center=[obstacles.spheres.centerX(k),obstacles.spheres.centerY(k),obstacles.spheres.centerZ(k)]; radius=obstacles.spheres.radius(k); [x,y,z]=sphere(50); surf(x*radius+center(1),y*radius+center(2),z*radius+center(3),'FaceColor',[0 0 1],'EdgeColor','none','FaceAlpha',pellucidity); end, end
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