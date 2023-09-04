function [centroids, y_pred] = KMeansClustering(train_x, train_y, nc, delta, distance)
% K-Means clustering algorithm implementation
% Inputs:
%   - train_x: input train data matrix with n rows (samples) and m columns (features)
%   - train_y: output train data matrix
%   - nc: number of clusters to be found by the algorithm
%   - delta: parameter
%   - distance: distance that will be performed

% Outputs:
%   - centroids: array of clusters
%   - y_pred: output estimated

disp('==============================');
disp('STARTS K-MEANS ALGORITHM');
disp('==============================');

[m,n] = size(train_x);

% Initialize cluster centers to random points
c = zeros(nc,n);
for i = 1:nc
    rnd = int16(rand*m + 1); % select a random vector from the input set
    c(i,:) = train_x(rnd,:); % assign this vector value to cluster (i)
end

n = 1000;
iter = 1;
while (iter < n)
% Determine the membership matrix U
% u(i,j) = 1 if euc_dist(x(j),c(i)) <= euc_dist(x(j),c(k)) for each k ~= i
% u(i,j) = 0 otherwise
for i = 1:nc
    for j = 1:m
        norma = normas(distance,train_x(j,:),c(i,:),train_x);
        d = norma^2;
        u(i,j) = 1;
        for k = 1:nc
            if k~=i
                norma = normas(distance,train_x(j,:),c(k,:),train_x);
                dd = norma^2;
                if dd < d
                    u(i,j) = 0;
                end
            end
        end
    end
end

% Compute the cost function J
J(iter) = 0;
for i = 1:nc
    JJ(i) = 0;
    for k = 1:m
        if u(i,k)==1
            norma = normas(distance,train_x(k,:),c(i,:),train_x);
            norma_cuad = norma^2;
            JJ(i) = JJ(i) + norma_cuad;
        end
    end
    J(iter) = J(iter) + JJ(i);
end

% Stop if either J is below a certain tolerance value,
% or its improvement over previous iteration is below a certain threshold
if (iter~=1) & (abs(J(iter-1) - J(iter)) < delta)
    break;
end

% Update the cluster centers
% c(i) = mean of all vectors belonging to cluster (i)
for i = 1:nc
    sum_x = 0;
    G(i) = sum(u(i,:));
    for k = 1:m
        if u(i,k)==1
            sum_x = sum_x + train_x(k,:);
        end
    end
    c(i,:) = sum_x ./ G(i);
end
iter = iter + 1;
end % while
centroids = c;
disp('Clustering Done.');


% ----------------- TESTING PHASE --------------------------
% Assign evaluation vectors to their respective clusters according
% to their distance from the cluster centers
for i = 1:nc
    for j = 1:m
        norma = normas(distance,train_x(j,:),c(i,:),train_x);
        d = norma^2;
        evu(i,j) = 1;
        for k = 1:nc
            if k~=i
                norma = normas(distance,train_x(j,:),c(k,:),train_x);
                dd = norma^2;
                if dd < d
                    evu(i,j) = 0;
                end
            end
        end
    end
end

% Analyze results
matrix_multiplication = 1:nc;
for i=1:nc
    posible_classes(i,:) = matrix_multiplication*evu;
    first_element = matrix_multiplication(1);
    matrix_multiplication = matrix_multiplication(2:end);
    matrix_multiplication(end+1) = first_element;
end


% calculate accuracy of each of the possible classes to select the correct one
max_correct = 0 ;
max_correct_classes_index = 1; 
for k=1:nc
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