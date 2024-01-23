function distance_matrix =  get_distance_matrix(distance, dataset, centroid_candidate)

[n, m] = size(dataset);
[r, s] = size(centroid_candidate);
distance_matrix = zeros(r,n);

for fila_ini = 1:r
    col = 1;
    for fila = 1:n
        datapoint1 = centroid_candidate(fila_ini,:);
        datapoint2 = dataset(fila,:);
        distance_matrix(fila_ini, col) = normas(distance, datapoint1, datapoint2, dataset);
        col = col + 1;
        %datapoint = dataset(fila,:);
        %distance_matrix(fila,1) = normas(distance, centroid_candidate, datapoint, dataset);
    end
end

%for fila_ini = 1:n
%    for fila = 1:n
%        norma = normas(distance, dataset(fila_ini, 1), dataset(fila, 2), dataset(:, 1:2));
%        distance_matrix(fila_ini, fila) = norma;
%    end
%end

end