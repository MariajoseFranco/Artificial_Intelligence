function [] = PlotCentroids(train_x, train_y, centroids, algorithm, dim)

if dim == 2
    figure
    gscatter(train_x(:,1), train_x(:,2), train_y,'rgb','osd');
    title(algorithm)
    xlabel('Sepal length');
    ylabel('Sepal width');
    hold on
    scatter(centroids(:,1),centroids(:,2),"black", 'x');
elseif dim == 3
    figure
    gscatter3(train_x(:,1), train_x(:,2),train_x(:,3), train_y);
    title(algorithm)
    xlabel('Sepal length');
    ylabel('Sepal width');
    zlabel('Petal length');
    hold on
    scatter3(centroids(:,1),centroids(:,2),centroids(:,3),"black", 'x');
else
    disp('No hay grafica para esta dimension de los datos')

end