function [centroids, y_pred] = MountainClustering(train_x, train_y, sigma, beta, gr, distance, data)
% Mountain clustering algorithm implementation
% Inputs:
%   - train_x: input train data matrix with n rows (samples) and m columns (features)
%   - train_y: output train data matrix
%   - sigma: parameter
%   - beta: parameter
%   - gr: granurality of the grid
%   - distance: distance that will be performed
%   - data: it determines if the dataset is real, toy, high dimension or
%   low dimension dataset

% Outputs:
%   - centroids: array of clusters
%   - y_pred: output estimated

disp('==============================');
disp('STARTS MOUNTAIN ALGORITHM');
disp('==============================');

[m, n] = size(train_x);
v_dim = gr*ones([1 n]);
cols = n;

%------------------------------------------------------------------
% First: setup a grid matrix of n-dimensions (V)
% (n = the dimension of input data vectors)
%------------------------------------------------------------------
M = zeros(v_dim);

%------------------------------------------------------------------
% Second: calculate the mountain function at every grid point
%------------------------------------------------------------------

% setup some aiding variables
cur = ones([1 n]);
for i = 1:n
    for j = 1:i
        cur(i) = cur(i)*v_dim(j);
    end
end
max_m = 0; % greatest density value
max_v = 0; % cluster center position
disp('Finding Cluster 1');

% loop over each grid point
for i = 1:cur(1,end)
    % calculate the vector indexes
    idx = i;
    for j = n:-1:2
        dim(j) = ceil(idx/cur(j-1));
        idx = idx - cur(j-1)*(dim(j)-1);
    end
    dim(1) = idx;
    % dim is holding the current point index vector
    % but needs to be normalized to the range [0,1]
    v = dim./gr;
    % calculate the mountain function for the current point
    M(i) = mountain(v,train_x,sigma,distance);
    if M(i) > max_m
        max_m = M(i);
        max_v = v;
        max_i = i;
    end
end

%------------------------------------------------------------------
% Third: select the first cluster center by choosing the point
% with the greatest density value
%------------------------------------------------------------------

c(1,:) = max_v;
ck = max_i;

%------------------------------------------------------------------
% FOR K CLUSTERS
%------------------------------------------------------------------

criterio_parada = 0;
k = 2;
while criterio_parada == 0
    Mnew = zeros(v_dim);
    max_m = 0;
    max_v = 0;
    str = sprintf('Finding Cluster %.0f%', k);
    disp(str);

    for i = 1:cur(1,end)
        % calculate the vector indexes
        idx = i;
        for j = n:-1:2
            dim(j) = ceil(idx/cur(j-1));
            idx = idx - cur(j-1)*(dim(j)-1);
        end
        dim(1) = idx;
        % dim is holding the current point index vector
        % but needs to be normalized to the range [0,1]
        v = dim./gr;
        
        % calculate the REVISED mountain function for the current point
        norma = normas(distance, v, c(k-1,:), train_x);
        norma_cuad = norma^2;
        % M_ck = mountain(c(k-1,:),train_x,sigma,distance);
        Mnew(i) = M(i) - M(ck)*exp(-norma_cuad/(2*beta^2));
        if Mnew(i) > max_m
            max_m = Mnew(i);
            max_v = v;
            max_i = i;
        end
    end
    if max_v == 0
        max_v = zeros(1,cols);
    end
    if ismember(max_v, c, 'rows')
        criterio_parada = 1;
    else
        c(k,:) = max_v;
        ck = max_i;
        M = Mnew;
        k = k + 1;
    end
end
centroids = c(1:end-1,:);

%------------------------------------------------------------------
% Evaluation
%------------------------------------------------------------------

% Assign every datapoint to its nearest cluster
for i = 1:size(centroids)
    for j = 1:m
        d = normas(distance, train_x(j,:), c(i,:), train_x);
        evu(i,j) = 1;
        for k = 1:size(centroids)
            if k~=i
                d2 = normas(distance, train_x(j,:), c(k,:), train_x);
                if d2 < d
                    evu(i,j) = 0;
                end
            end
        end
    end
end

% Analyze results
matrix_multiplication = 1:size(centroids);
for i=1:size(centroids)
    posible_classes(i,:) = matrix_multiplication*evu;
    first_element = matrix_multiplication(1);
    matrix_multiplication = matrix_multiplication(2:end);
    matrix_multiplication(end+1) = first_element;
end


% calculate accuracy of each of the possible classes to select the correct one
max_correct = 0 ;
max_correct_classes_index = 1; 
for k=1:size(centroids)
    correct = 0;
    for i = 1:m
        if posible_classes(k,i) == train_y(i)
            correct = correct + 1;
        end
    end
    if correct> max_correct
        max_correct = correct;
        max_correct_classes_index = k; 
    end
end

accuracy = max_correct/m;
str = sprintf('Accuracy: %f', accuracy);
disp(str);
y_pred = posible_classes(max_correct_classes_index, :) ;
y_pred = y_pred.';

end


function m = mountain(centroid_candidate, train_x, sigma, distance)

distance_matrix = get_distance_matrix(distance, train_x, centroid_candidate);
distance_matrix_cuad = distance_matrix.^2;
m = sum(exp((-distance_matrix_cuad)/(2*sigma^2)));

end