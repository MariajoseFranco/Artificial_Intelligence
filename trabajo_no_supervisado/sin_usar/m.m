% generate some random data
% cluster the data using DBSCAN with eps=0.5 and minPts=5
algorithm = 'DBSCAN Clustering';
data = train_x;
[centroids_D, y_pred_D] = DBSCANClustering(data, 0.35, 15,'euc');

% plot the clusters and noise points
PlotCentroids(train_x, y_pred_D, centroids_D, algorithm, dim)
disp('CENTROIDS:');
disp(centroids_D);
