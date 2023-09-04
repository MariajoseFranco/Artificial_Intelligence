function [train_x, train_y] = DataProcessing(real)

[train_x, train_y] = LoadData(real);
train_x = NormalizeData(train_x);

end

function [data, y] = LoadData(real)

if real == 'r'
    data = xlsread('Rice_Cammeo_Osmancik.xlsx');
    data = data(randperm(size(data,1)),:);
    y = data(1:300,end);
    data = data(1:300,1:4);
elseif real == 'h'
    data = xlsread('hd.xlsx');
    data = data(randperm(size(data,1)),:);
    y = data(1:300,end);
    data = data(1:300,1:end-2);
elseif real == 'l'
    data = xlsread('ld.xlsx');
    data = data(randperm(size(data,1)),:);
    y = data(1:300,end);
    data = data(1:300,1:end-2);
elseif real == 't'
    load fisheriris;
    data = meas;
    output = array2table(species);
    [~, ~, labels] = unique(output);
    labels = array2table(labels);
    labels = labels{:,:};
    data(:, end+1) = labels;
    y = data(:,end);
    
    % Para 2 dimensiones
    %d = data(:,1:2);
    %d(:,end+1) = data(:,end);
    %data = d;
    
    % Para 3 dimensiones
    %d = data(:,1:3);
    %d(:,end+1) = data(:,end);
    %data = d;
end

end

function [train_x, train_y, test_x, test_y] = SplitData(dataset)

[m,n] = size(dataset);
train_proportion = 0.80;

rand_index = transpose(randperm(m));
train_size = round(train_proportion*m);
training_dataset = dataset(rand_index(1:train_size,:), :);
train_x = training_dataset(:,1:end-1);
train_y = training_dataset(:,end);

testing_dataset = dataset(rand_index(train_size+1:end), :);
test_x = testing_dataset(:,1:end-1);
test_y = testing_dataset(:,end);

end

function data_normalized = NormalizeData(dataset)

% X
[m, n] = size(dataset);
data_normalized = zeros([m, n]);
for i = 1:n
    minimo = min(dataset(:,i));
    maximo = max(dataset(:,i));
    data_normalized(:,i) = (dataset(:,i)-minimo)/(maximo-minimo);
end

% Y
% [m, n] = size(y);
% y_normalized = zeros([m, n]);
% for i = 1:n
%     minimo = min(y(:,i));
%     maximo = max(y(:,i));
%     y_normalized(:,i) = (y(:,i)-minimo)/(maximo-minimo);
% end

end
