function [centroids, y_pred] = DBSCANClustering(train_x, train_y, eps, minPts, distance)
% DBSCAN clustering algorithm implementation
% Inputs:
%   - data: input data matrix with n rows (samples) and m columns (features)
%   - eps: radius of the neighborhood of a data point
%   - minPts: minimum number of points required to form a dense region
% Outputs:
%   - clusters: cell array of clusters, each containing the indices of the corresponding data points
%   - noise: indices of the noise points

disp('==============================');
disp('STARTS DBSCAN ALGORITHM');
disp('==============================');

% initialize variables
[n,m] = size(train_x);
visited = false(n, 1);
clusters = 1;
labels = (0)*ones(n,1);
    
% compute pairwise distances between data points
D = get_distance_matrix(distance,train_x, train_x);

for i = 1:n
    if ~visited(i)
        visited(i) = 1;
        nbrs = find(D(i,:) <= eps);
        if length(nbrs) >= minPts
            centroids(i) = 1;
            for j = 1:length(nbrs)
                nbr = nbrs(j);
                if ~visited(nbr)
                    visited(nbr) = 1;
                end
                if labels(nbr) == 0
                    labels(nbr) = clusters;
                end
            end
            clusters = clusters + 1;
        end
    end
end

coord_centroid = find(centroids == 1);
lc = length(coord_centroid);
centroids = zeros(lc,m);
for i = 1:lc
    coord = coord_centroid(i);
    centroids(i,:) = train_x(coord,:);
end

% Calculate accuracy
correct = 0;
for i = 1:length(labels)
    if labels(i) == train_y(i)
        correct = correct + 1;
    end
end

accuracy = correct/n;
str = sprintf('Accuracy: %f', accuracy);
disp(str);
y_pred = labels;

end