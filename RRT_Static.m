%% 

function RRT_3D_Adjustable_LineWidth()

    clear;
    clc;
    close all;

    %% 
    space.x_min = 0; space.x_max = 1000;
    space.y_min = 0; space.y_max = 1000;
    space.z_min = 0; space.z_max = 1000;
    
    start_point = [0, 0, 0];
    goal_point = [1000, 1000, 1000];
    
    params.step_size = 10;
    params.max_iter = 3000;
    params.goal_threshold = 10;
    params.goal_bias = 0.02;

    obstacles.cubes = createCubeObject();
    obstacles.cylinders = createCylinderObject();
    obstacles.spheres = createSphereObject();

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

    fprintf('3D RRT planning started (with goal bias)...\n');
    path_found = false;
    for iter = 1:params.max_iter
        rand_point = sample_point_3d_biased(space, goal_point, params.goal_bias); 
        [near_node, near_idx] = find_nearest_node(rand_point, T);
        new_node = expand_node(near_node, rand_point, params.step_size);
        if is_collision_3d(near_node, new_node, obstacles), continue; end
        T.nodes(end+1, :) = new_node;
        T.parent(end+1) = near_idx;
        

        plot3([near_node(1), new_node(1)], [near_node(2), new_node(2)], [near_node(3), new_node(3)], 'b-', 'LineWidth', 1.5);

        drawnow limitrate;
        
        if pdist([new_node; goal_point], 'euclidean') < params.goal_threshold
            fprintf('Goal successfully found! Iterations: %d\n', iter);
            path_found = true;
            break;
        end
    end

    %% 
    if path_found
        final_path = reconstruct_path(T, goal_point);
        
        for k = 1:size(final_path, 1) - 1

            line([final_path(k,1), final_path(k+1,1)], ...
                 [final_path(k,2), final_path(k+1,2)], ...
                 [final_path(k,3), final_path(k+1,3)], ...
                 'LineWidth', 3, 'Color', 'red'); 
        end
    else
        fprintf('Planning failed. Maximum iterations reached.\n');
    end
    hold off;
end

%% 
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
function draw_obstacles(obstacles)
    pellucidity = 0.3;
    if obstacles.cubes.exist, for k = 1:length(obstacles.cubes.axisX), origin = [obstacles.cubes.axisX(k), obstacles.cubes.axisY(k), obstacles.cubes.axisZ(k)]; edges = [obstacles.cubes.length(k), obstacles.cubes.width(k), obstacles.cubes.height(k)]; plotcube(edges, origin, pellucidity, [1 1 0]); end, end
    if obstacles.cylinders.exist, for k = 1:length(obstacles.cylinders.X), center = [obstacles.cylinders.X(k), obstacles.cylinders.Y(k), obstacles.cylinders.Z(k)]; radius = obstacles.cylinders.radius(k); height = obstacles.cylinders.height(k); [x, y, z] = cylinder(radius, 30); z = z * height + center(3); surf(x + center(1), y + center(2), z, 'FaceColor', [0 1 0], 'EdgeColor', 'none', 'FaceAlpha', pellucidity); fill3(x(1,:) + center(1), y(1,:) + center(2), z(1,:), [0 1 0], 'FaceAlpha', pellucidity, 'EdgeColor', 'none'); fill3(x(2,:) + center(1), y(2,:) + center(2), z(2,:), [0 1 0], 'FaceAlpha', pellucidity, 'EdgeColor', 'none'); end, end
    if obstacles.spheres.exist, for k = 1:length(obstacles.spheres.centerX), center = [obstacles.spheres.centerX(k), obstacles.spheres.centerY(k), obstacles.spheres.centerZ(k)]; radius = obstacles.spheres.radius(k); [x, y, z] = sphere(50); surf(x*radius+center(1), y*radius+center(2), z*radius+center(3), 'FaceColor', [0 0 1], 'EdgeColor', 'none', 'FaceAlpha', pellucidity); end, end
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