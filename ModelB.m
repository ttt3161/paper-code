function RRT_Ablation_Halton_GoalBias_NoSmooth_3D_50Runs()

    clear;
    clc;
    close all;

 
    space.x_min = 0; space.x_max = 1000;
    space.y_min = 0; space.y_max = 1000;
    space.z_min = 0; space.z_max = 1000;
    
    start_point = [0, 0, 0];
    goal_point = [1000, 1000, 1000];
    
    rrt_params.goal_bias = 0.1; 
    rrt_params.max_iter = 3000; 
    rrt_params.goal_threshold = 10;
    rrt_params.step_size = 20; 


    fig = figure; hold on;
    axis([space.x_min, space.x_max, space.y_min, space.y_max, space.z_min, space.z_max]);
    axis equal; view(3); grid on;
    
    ax = gca; ax.FontSize = 16;
    xlabel('X'); ylabel('Y'); zlabel('Z');

    TOTAL_RUNS = 50;
    stats_iter = [];
    stats_time = [];
    stats_path_length = [];
    stats_nodes = [];
    stats_utilization = [];
    success_count = 0;


    fprintf('====== Starting %d independent repeated experiments ======\n', TOTAL_RUNS);
    fprintf('       (Ablation: Halton + Goal Bias + No Smooth/APF)       \n');
    fprintf('==========================================================\n');

    for run_idx = 1:TOTAL_RUNS
        fprintf('\n--- Running experiment %d/%d ---\n', run_idx, TOTAL_RUNS);
        
        cla;
        scatter3(start_point(1), start_point(2), start_point(3), 100, 'go', 'MarkerFaceColor', 'g');
        scatter3(goal_point(1), goal_point(2), goal_point(3), 100, 'ro', 'MarkerFaceColor', 'r');
        h_final_path = plot3(NaN,NaN,NaN,'r-', 'LineWidth', 2.5);

        obstacles.cubes = createCubeObject();
        obstacles.cylinders = createCylinderObject();
        obstacles.spheres = createSphereObject();

        planning_timer = tic; 
        
        [T, iter_used] = find_initial_path_rrt_halton_3d(space, start_point, goal_point, rrt_params, obstacles);
        

        planning_time = toc(planning_timer);


        if isempty(T)
            fprintf('-> Failed! Maximum iterations reached.\n');
            continue; 
        end
        
        success_count = success_count + 1;
        original_path = reconstruct_path(T, goal_point);
        final_used_path = original_path;
        
        set(h_final_path, 'XData', final_used_path(:,1), 'YData', final_used_path(:,2), 'ZData', final_used_path(:,3));
        drawnow;

        total_nodes = size(T.nodes, 1);
        path_length = 0;
        for k = 1:size(final_used_path, 1) - 1
            path_length = path_length + norm(final_used_path(k+1,:) - final_used_path(k,:));
        end
        
        utilization_rate = (size(original_path, 1) / total_nodes) * 100;
        
        stats_iter = [stats_iter; iter_used];
        stats_time = [stats_time; planning_time];
        stats_path_length = [stats_path_length; path_length];
        stats_nodes = [stats_nodes; total_nodes];
        stats_utilization = [stats_utilization; utilization_rate];
        
        fprintf('-> Success! Iterations: %d, Time: %.4f s, Length: %.2f\n', iter_used, planning_time, path_length);
    end


    fprintf('\n==================================\n');
    fprintf('   %d Independent Runs Average Statistics\n', TOTAL_RUNS);
    fprintf('==================================\n');
    fprintf('Success Rate:                    %d / %d (%.2f %%)\n', success_count, TOTAL_RUNS, (success_count/TOTAL_RUNS)*100);
    
    if success_count > 0
        fprintf('Average Iterations:              %.2f\n', mean(stats_iter)); 
        fprintf('Average Planning Time:           %.4f s\n', mean(stats_time));
        fprintf('Average Path Length:             %.2f\n', mean(stats_path_length));
        fprintf('Average Total Nodes Generated:   %.2f\n', mean(stats_nodes));
        fprintf('Average Node Utilization Rate:   %.2f %%\n', mean(stats_utilization));
    else
        fprintf('All experiments failed. No statistics available.\n');
    end
    fprintf('==================================\n\n');
 
    draw_obstacles(obstacles);
    hold off;
end



function [T, iter] = find_initial_path_rrt_halton_3d(space, start_point, goal_point, rrt_params, obstacles)
    T.nodes(1, :) = start_point; T.parent(1) = 0;
    try, halton_stream = haltonset(3); halton_stream = scramble(halton_stream, 'RR2'); 
    catch, error('Please install the Statistics and Machine Learning Toolbox™.'); end
    
    for iter = 1:rrt_params.max_iter

        rand_point = sample_point_halton_3d(space, goal_point, rrt_params.goal_bias, halton_stream, iter);
        [near_node, near_idx] = find_nearest_node(rand_point, T);
        new_node = expand_node_standard_3d(near_node, rand_point, rrt_params.step_size);

        if is_collision_3d(near_node, new_node, obstacles), continue; end
        
        T.nodes(end+1, :) = new_node; T.parent(end+1) = near_idx;
        

        if pdist([new_node; goal_point], 'euclidean') < rrt_params.goal_threshold, return; end
    end
    T = [];
end

function new_node = expand_node_standard_3d(near_n, rand_p, step_size)
    direction_vec = rand_p - near_n; 
    dist = norm(direction_vec); 
    if dist < step_size
        new_node = rand_p; 
    else
        unit_vec = direction_vec / dist; 
        new_node = near_n + step_size * unit_vec; 
    end
end

function rand_p = sample_point_halton_3d(space, goal_p, goal_bias, halton_stream, iter)
    if rand < goal_bias, rand_p = goal_p;
    else, p_unit = halton_stream(iter, :); rand_p(1) = p_unit(1) * (space.x_max - space.x_min) + space.x_min; rand_p(2) = p_unit(2) * (space.y_max - space.y_min) + space.y_min; rand_p(3) = p_unit(3) * (space.z_max - space.z_min) + space.z_min; end
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