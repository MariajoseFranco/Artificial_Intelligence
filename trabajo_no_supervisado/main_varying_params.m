% Global Variables
distance = 'euc';
real_data = 'r';
hd_data = 'h';
ld_data = 'l';
toy_data = 't';
data = hd_data;

% Load, split and normalize data
[train_x, train_y] = DataProcessing(data);
[~, dim] = size(train_x);
PlotReal(train_x, train_y, dim);

% Mountain Algorithm
algorithm = 'Mountain Clustering';
% sigma = 0.1;
% beta = 0.4;
gr = 20;

for sigma = 0.1:0.2:0.5
    for beta = 0.2:0.2:0.6
        [centroids_M, y_pred_M] = MountainClustering(train_x, train_y, sigma, beta, gr, distance, data);
        PlotCentroids(train_x, y_pred_M, centroids_M, algorithm, dim);
        eval = evalclusters(train_x,y_pred_M,'silhouette')
        disp('sigma:');
        disp(sigma);
        disp('beta:');
        disp(beta);
        disp('gr:');
        disp(gr);
        disp('CENTROIDS:');
        disp(centroids_M);
    end
end


% Substractive Algorithm
algorithm = 'Substractive Clustering';
% ra = 0.8;
% rb = 2*ra;
for ra = 0.6:0.2:1
    for p = 1:0.5:2
        rb = p*ra;
        [centroids_S, y_pred_S] = SubtractiveClustering(train_x, train_y, ra, rb, distance, data);
        PlotCentroids(train_x, y_pred_S, centroids_S, algorithm, dim)
        eval = evalclusters(train_x,y_pred_S,'silhouette')
        disp('ra');
        disp(ra);
        disp('rb');
        disp(rb);
        disp('CENTROIDS:');
        disp(centroids_S);
        [nc, ~] = size(centroids_S);
    end
end

% K-Means Algorithm
algorithm = 'K-Means Clustering';
delta = 1e-5;
for nc = 1:9
    [centroids_KM, y_pred_KM] = KMeansClustering(train_x, train_y, nc, delta, distance);
    PlotCentroids(train_x, y_pred_KM, centroids_KM, algorithm, dim)
    eval = evalclusters(train_x,y_pred_KM,'silhouette')
    disp('nc');
    disp(nc);
    disp('CENTROIDS:');
    disp(centroids_KM);
end

% Fuzzy C-Means Algorithm
algorithm = 'Fuzzy C-Means Clustering';
delta = 1e-5;
% m_exp = 7;
for m_exp = 3:2:7
    for nc = 1:3
        [centroids_CM, y_pred_CM] = FuzzyCMeansClustering(train_x, train_y, nc, m_exp, delta, distance);
        PlotCentroids(train_x, y_pred_CM, centroids_CM, algorithm, dim)
        eval = evalclusters(train_x,y_pred_CM,'silhouette')
        disp('m_exp');
        disp(m_exp);
        disp('nc');
        disp(nc);
        disp('CENTROIDS:');
        disp(centroids_CM);
    end
end

% DBSCAN
algorithm = 'DBSCAN Clustering';
% eps = 0.75;
% minPts = 10;
for eps = 0.25:0.25:0.75
    for minPts = 10:10:30
        [centroids_D, y_pred_D] = DBSCANClustering(train_x, train_y, eps, minPts, distance);
        PlotCentroids(train_x, y_pred_D, centroids_D, algorithm, dim)
        %eval = evalclusters(train_x,y_pred_D,'silhouette')
        disp('eps');
        disp(eps);
        disp('minPts');
        disp(minPts);
        disp('CENTROIDS:');
        disp(centroids_D);
    end
end
