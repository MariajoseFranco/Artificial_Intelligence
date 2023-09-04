function [clusters, tree] = hac(data, method, distance)
% Hierarchical Agglomerative Clustering (HAC) algorithm implementation
% Inputs:
%   - data: input data matrix with n rows (samples) and m columns (features)
%   - method: linkage method ('single', 'complete', or 'average')
% Outputs:
%   - clusters: cell array of clusters, each containing the indices of the corresponding data points
%   - tree: clustering tree, represented as a matrix with n-1 rows and 3 columns

    % initialize variables
    n = size(data, 1);
    dd = get_distance_matrix(distance, data, data);
    tree = zeros(n-1, 3);
    clusters = num2cell(1:n);
    v = ones(1,n)*1000;
    m = size(dd,1);
    dd(1:(m+1):end) = v;
    
    % start clustering
    for i = 1:n-1
        % find the two closest clusters
        [minDist, minIndex] = min(dd);
        [row, col] = ind2sub([n-i+1, n-i+1], minIndex);
        c1 = clusters{row};
        c2 = clusters{col};
        
        % update the clustering tree
        tree(i,:) = [c1(1), c2(1), minDist];
        
        % merge the two closest clusters
        clusters{row} = [c1, c2];
        clusters(col) = [];
        
        % update distances between new cluster and remaining clusters
        switch method
            case 'single'
                % single linkage method
                dd(row,:) = min(dd(row,:), dd(col,:));
                dd(:,row) = dd(row,:)';
                dd(col,:) = inf;
                dd(:,col) = inf;
            case 'complete'
                % complete linkage method
                dd(row,:) = max(dd(row,:), dd(col,:));
                dd(:,row) = dd(row,:)';
                dd(col,:) = inf;
                dd(:,col) = inf;
            case 'average'
                % average linkage method
                n1 = length(c1);
                n2 = length(c2);
                dd(row,:) = (n1*dd(row,:) + n2*dd(col,:)) / (n1+n2);
                dd(:,row) = dd(row,:)';
                dd(col,:) = inf;
                dd(:,col) = inf;
        end
    end
    
    % convert tree to linkage format
    tree(:,1:2) = sort(tree(:,1:2), 2);
    tree = [tree(:,2), tree(:,1), tree(:,3)];
end
