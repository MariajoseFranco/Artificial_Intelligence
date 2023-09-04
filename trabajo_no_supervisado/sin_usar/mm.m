% generate some random data
data = train_x(:,1:2);

% cluster the data using HAC with complete linkage
[clusters, tree] = hac(data, 'complete', distance);

% plot the dendrogram
figure;
dendrogram(tree, 'ColorThreshold', 2, 'Orientation', 'top');
title('HAC Clustering with Complete Linkage');

% plot the clusters
figure;
scatter(data(:,1), data(:,2), 20, 'k');
hold on;
for i = 1:length(clusters)
    scatter(data(clusters{i},1), data(clusters{i},2), 20, rand(1,3), 'filled');
end
title('HAC Clustering with Complete Linkage');
