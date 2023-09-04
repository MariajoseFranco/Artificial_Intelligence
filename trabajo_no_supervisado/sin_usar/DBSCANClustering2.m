function [clusters, labels] = DBSCANClustering(data, eps, minPts, distance)
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
[n,m] = size(data);
visited = false(n, 1);
noise = [];
clusters = zeros(1);
nbrs = cell(n, 1);
labels = (-1)*ones(n,1);
labels_clusters = 1;
    
% compute pairwise distances between data points
D = get_distance_matrix(distance,data, data);

% find neighbors of each data point within eps distance
for i = 1:n
    nbrs{i} = find(D(i,:) <= eps);
end

% start clustering
for i = 1:n
    if visited(i)
        continue;
    end
    
    visited(i) = true;
    nbr_i = nbrs{i};
    
    if length(nbr_i) < minPts
        % mark as noise point
        noise(end+1) = i;
    else
        % create a new cluster
        clusters(labels_clusters,1) = [i];
        c = length(clusters(labels_clusters));
        
        % expand the cluster
        for j = 1:length(clusters(c))
            while j <= length(clusters(c))
                p = clusters(c,j);
                visited(p) = true;
                nbr_p = nbrs{p};
                
                if length(nbr_p) >= minPts
                    % add new points to the cluster
                    for k = 1:length(nbr_p)
                        q = nbr_p(k);
                        if ~visited(q)
                            visited(q) = true;
                        end
                        if labels(q) == -1
                            labels(q) = labels_clusters;
                            %nbr_q = nbrs{q};
                            %if length(nbr_q) >= minPts
                            %    nbr_p = union(nbr_p, nbr_q);
                            %end
                        end     
                        if labels_clusters == 1
                            if isempty(find([clusters] == q, 1))
                                clusters(labels_clusters,k) = q;
                            end
                        end
                    end
                    labels_clusters =  labels_clusters + 1;
                end         
                j = j + 1;
            end
        end
    end
end
