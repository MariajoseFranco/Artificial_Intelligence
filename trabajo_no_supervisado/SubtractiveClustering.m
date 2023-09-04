function [centroids, y_pred] = SubtractiveClustering(train_x, train_y, ra, rb, distance, data)
% Subtractive clustering algorithm implementation
% Inputs:
%   - train_x: input train data matrix with n rows (samples) and m columns (features)
%   - train_y: output train data matrix
%   - ra: parameter
%   - rb: parameter
%   - distance: distance that will be performed
%   - data: it determines if the dataset is real, toy, high dimension or
%   low dimension dataset

% Outputs:
%   - centroids: array of clusters
%   - y_pred: output estimated

disp('================================');
disp('STARTS SUBSTRACTIVE ALGORITHM');
disp('================================');

%------------------------------------------------------------------
% First: Initialize the density matrix and some variables
%------------------------------------------------------------------

[m, n] = size(train_x);
D = zeros([m 1]);
cols = n;

%------------------------------------------------------------------
% Second: calculate the density function at every data point
%------------------------------------------------------------------

% setup some aiding variables
max_d = 0; % greatest density value
max_x = 0; % cluster center position
disp('Finding Cluster 1');
% loop over each data point
for i = 1:m
    % calculate the density function for the current point
    D(i) = density(train_x(i,:),train_x,ra,distance);
    if D(i) > max_d
        max_d = D(i);
        max_x = train_x(i,:);
        max_i = i;
    end
end

%------------------------------------------------------------------
% Third: select the first cluster center by choosing the point
% with the greatest density value
%------------------------------------------------------------------

c(1,:) = max_x;
ck = max_i;

%------------------------------------------------------------------
% FOR K CLUSTERS
%------------------------------------------------------------------

criterio_parada = 0;
k = 2;
highest = D(ck);
while criterio_parada == 0
    Dnew = zeros([m 1]);
    max_d = 0;
    max_x = 0;
    str = sprintf('Finding Cluster %.0f% ', k);
    disp(str);

    for i = 1:m
        % calculate the REVISED density function for the current point
        norma = normas(distance, train_x(i,:), c(k-1,:), train_x);
        norma_cuad = norma^2;
        Dnew(i) = D(i) - highest*exp((-norma_cuad)/((rb/2)^2));
        if Dnew(i) > max_d
            max_d = Dnew(i);
            max_x = train_x(i,:);
            max_i = i;
        end
    end
    if max_x == 0
        max_x = zeros(1,cols);
    end
    if ismember(max_x, c, 'rows')
        criterio_parada = 1;
    else
        c(k,:) = max_x;
        ck = max_i;
        D = Dnew;
        k = k + 1;
    end
end
if data ~= 't'
    centroids = c(1:end-1,:);
else
    centroids = c;
end

%------------------------------------------------------------------
% Evaluation
%------------------------------------------------------------------

% Assign every test vector to its nearest cluster
for i = 1:size(centroids)
    for j = 1:m
        d = normas(distance, train_x(j,:), c(i,:), train_x);
        evu(i,j) = 1;
        for k=1:size(centroids)
            if k~=i
                d2 = normas(distance, train_x(j,:), c(k,:), train_x);
                if d2 < d
                    evu(i,j) = 0;
                end
            end
        end
    end
end

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

function d = density(centroid_candidate, train_x, ra, distance)

distance_matrix = get_distance_matrix(distance, train_x, centroid_candidate);
distance_matrix_cuad = distance_matrix.^2;
d = sum(exp((-distance_matrix_cuad)/((ra/2)^2)));

end