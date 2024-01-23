% Global Variables
distance = 'euc';
real_data = 'r';
hd_data = 'h';
ld_data = 'l';
toy_data = 't';
data = ld_data;

% Load, split and normalize data
[train_x, train_y] = DataProcessing(data);
[~, dim] = size(train_x);
PlotReal(train_x, train_y, dim);

% Mountain Algorithm
algorithm = 'Mountain Clustering';
sigma = 0.1;
beta = 0.6;
gr = 20;

[centroids_M, y_pred_M] = MountainClustering(train_x, train_y, sigma, beta, gr, distance, data);
PlotCentroids(train_x, y_pred_M, centroids_M, algorithm, dim);
eval = evalclusters(train_x,y_pred_M,'silhouette')
disp('CENTROIDS:');
disp(centroids_M);

% Substractive Algorithm
algorithm = 'Substractive Clustering';
ra = 0.6;
rb = 2*ra;

[centroids_S, y_pred_S] = SubtractiveClustering(train_x, train_y, ra, rb, distance, data);
PlotCentroids(train_x, y_pred_S, centroids_S, algorithm, dim)
eval = evalclusters(train_x,y_pred_S,'silhouette')
disp('CENTROIDS:');
disp(centroids_S);
[nc, ~] = size(centroids_S);

% % K-Means Algorithm
algorithm = 'K-Means Clustering';
delta = 1e-5;

[centroids_KM, y_pred_KM] = KMeansClustering(train_x, train_y, nc, delta, distance);
PlotCentroids(train_x, y_pred_KM, centroids_KM, algorithm, dim)
eval = evalclusters(train_x,y_pred_KM,'silhouette')
disp('CENTROIDS:');
disp(centroids_KM);


% Fuzzy C-Means Algorithm
algorithm = 'Fuzzy C-Means Clustering';
delta = 1e-5;
m_exp = 3;

[centroids_CM, y_pred_CM] = FuzzyCMeansClustering(train_x, train_y, nc, m_exp, delta, distance);
PlotCentroids(train_x, y_pred_CM, centroids_CM, algorithm, dim)
eval = evalclusters(train_x,y_pred_CM,'silhouette')
disp('CENTROIDS:');
disp(centroids_CM);

% DBSCAN
algorithm = 'DBSCAN Clustering';
eps = 0.75;
minPts = 20;

[centroids_D, y_pred_D] = DBSCANClustering(train_x, train_y, eps, minPts, distance);
PlotCentroids(train_x, y_pred_D, centroids_D, algorithm, dim)
eval = evalclusters(train_x,y_pred_D,'silhouette')
disp('CENTROIDS:');
disp(centroids_D);
