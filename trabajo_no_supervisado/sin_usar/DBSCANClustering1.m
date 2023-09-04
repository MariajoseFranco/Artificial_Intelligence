function [coord_centroid, labels] = DBSCANClustering1(train_x, eps, minPts, distance)
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
labels = (-1)*ones(n,1);
    
% compute pairwise distances between data points
D = get_distance_matrix(distance,train_x, train_x);

for i = 1:n
    if ~visited(i)
        visited(i) = 1;
        nbrs = find(D(i,:) <= eps);
        if length(nbrs) >= minPts
            centroids(i) = 1;
            for j = 1:length(nbrs)
                if ~visited(j)
                    visited(j) = 1;
                end
                if labels(j) == -1
                    labels(j) = clusters;
                end
                clusters = clusters + 1;
            end
        end
    end
end

coord_centroid = train_x(centroids == 1);


end