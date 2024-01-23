function [] = PlotReal(train_x, train_y, dim)

if dim == 2
    figure
    gscatter(train_x(:,1), train_x(:,2), train_y,'rgb','osd');
    title('Real Data');
    xlabel('Sepal length');
    ylabel('Sepal width');
elseif dim == 3
    figure
    gscatter3(train_x(:,1), train_x(:,2),train_x(:,3), train_y);
    title('Real Data');
    xlabel('Sepal length');
    ylabel('Sepal width');
    zlabel('Petal length');
else
    disp('No hay grafica para esta dimension de los datos')

end