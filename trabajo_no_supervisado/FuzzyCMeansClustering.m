function [centroids, y_pred] = FuzzyCMeansClustering(train_x, train_y, nc, m_exp, delta, distance)
% Fuzzy C-Means clustering algorithm implementation
% Inputs:
%   - train_x: input train data matrix with n rows (samples) and m columns (features)
%   - train_y: output train data matrix
%   - nc: number of clusters to be found by the algorithm
%   - m_exp: parameter
%   - delta: parameter
%   - distance: distance that will be performed

% Outputs:
%   - centroids: array of clusters
%   - y_pred: output estimated

disp('==============================');
disp('STARTS FUZZY C-MEANS ALGORITHM');
disp('==============================');

[m,n] = size(train_x);

% Initialize the membership matrix with random values between 0 and 1
% such that the summation of membership degrees for each vector equals unity
u = zeros(nc,m);
for i = 1:m
    r = rand(1, 3);
    r = r / sum(r);
    u(1,i) = r(1);
    u(2,i) = r(2);
    u(3,i) = r(3);
end

prevJ = 0;
J = 0;
n = 1000;
iter = 1;
while (iter < n)
% Calculate the fuzzy cluster centers
for i = 1:nc
    sum_ux = 0;
    sum_u = 0;
    for j = 1:m
        sum_ux = sum_ux + (u(i,j)^m_exp)*train_x(j,:);
        sum_u = sum_u + (u(i,j)^m_exp);
    end
    c(i,:) = sum_ux ./ sum_u;
end

% Compute the cost function J
J(iter) = 0;
for i = 1:nc
    JJ(i) = 0;
    for j = 1:m
        norma = normas(distance,train_x(j,:),c(i,:), train_x);
        norma_cuad = norma^2;
        JJ(i) = JJ(i) + (u(i,j)^m_exp)*norma_cuad;
    end
    J(iter) = J(iter) + JJ(i);
end

% Stop if either J is below a certain tolerance value,
% or its improvement over previous iteration is below a certain threshold
if (iter~=1) & (abs(J(iter-1) - J(iter)) < delta)
    break;
end

% Update the membership matrix U
for i = 1:nc
    for j = 1:m
        sum_d = 0;
        for k = 1:nc
            norma = normas(distance,c(i,:),train_x(j,:),train_x);
            d = norma^2;
            norma = normas(distance,c(k,:),train_x(j,:),train_x);
            dd = norma^2;
            sum_d = sum_d + (d/dd)^(2/(m_exp-1));
        end
        u(i,j) = 1/sum_d;
    end
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
        sum_d = 0;
        for k = 1:nc
            norma = normas(distance,c(i,:),train_x(j,:),train_x);
            d = norma^2;
            norma = normas(distance,c(k,:),train_x(j,:),train_x);
            dd = norma^2;
            sum_d = sum_d + (d/dd)^(2/(m_exp-1));
        end
        evu(i,j) = 1/sum_d;
    end
end

% defuzzify the membership matrix
for j = 1:m
    for i=1:nc
        maximum = max(evu(:,j));
        if evu(i,j) == maximum
            evu(i,j) = 1;
        else
            evu(i,j) = 0;
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