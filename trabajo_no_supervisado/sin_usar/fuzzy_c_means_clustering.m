% Fuzzy C-means clustering
% ------------------- CLUSTERING PHASE -------------------
% Load the Training Set
TrSet = load('TrainingSet.txt');
[m,n] = size(TrSet); % (m samples) x (n dimensions)
for i = 1:m % the output (last column) values (0,1,2,3) are mapped to (0,1)
    if TrSet(i,end)>=1
        TrSet(i,end)=1;
    end
end

% find the range of each attribute (for normalization later)
for i = 1:n
    range(1,i) = min(TrSet(:,i));
    range(2,i) = max(TrSet(:,i));
end
x = Normalize(TrSet, range); % normalize the data set to a hypercube
x(:,end) = []; % get rid of the output column
[m,n] = size(x);
nc = 2; % number of clusters = 2

% Initialize the membership matrix with random values between 0 and 1
% such that the summation of membership degrees for each vector equals unity
u = zeros(nc,m);
for i = 1:m
    u(1,i) = rand;
    u(2,i) = 1 - u(1,i);
end

% Clustering Loop
m_exp = 12;
prevJ = 0;
J = 0;
delta = 1e-5;
n = 1000;
iter = 1;
while (iter < n)
% Calculate the fuzzy cluster centers
for i = 1:nc
    sum_ux = 0;
    sum_u = 0;
    for j = 1:m
        sum_ux = sum_ux + (u(i,j)^m_exp)*x(j,:);
        sum_u = sum_u + (u(i,j)^m_exp);
    end
    c(i,:) = sum_ux ./ sum_u;
end

% Compute the cost function J
J(iter) = 0;
for i = 1:nc
    JJ(i) = 0;
    for j = 1:m
        JJ(i) = JJ(i) + (u(i,j)^m_exp)*euc_dist(x(j,:),c(i,:));
    end
    J(iter) = J(iter) + JJ(i);
end

% Stop if either J is below a certain tolerance value,
% or its improvement over previous iteration is below a certain threshold
str = sprintf('iteration: %.0d, J=%d', iter, J);
disp(str);
if (iter~=1) & (abs(J(iter-1) - J(iter)) < delta)
    break;
end

% Update the membership matrix U
for i = 1:nc
    for j = 1:m
        sum_d = 0;
        for k = 1:nc
            sum_d = sum_d + (euc_dist(c(i,:),x(j,:))/euc_dist(c(k,:),x(j,:)))^(2/(m_exp-1));
        end
        u(i,j) = 1/sum_d;
    end
end
iter = iter + 1;
end % while
disp('Clustering Done.');


% ----------------- TESTING PHASE --------------------------
% Load the evaluation data set
EvalSet = load('EvaluationSet.txt');
[m,n] = size(EvalSet);
for i = 1:m
    if EvalSet(i,end)>=1
        EvalSet(i,end)=1;
    end
end
x = Normalize(EvalSet, range);
x(:,end) = [];
[m,n] = size(x);

% Assign evaluation vectors to their respective clusters according
% to their distance from the cluster centers
for i = 1:nc
    for j = 1:m
        sum_d = 0;
        for k = 1:nc
            sum_d = sum_d + (euc_dist(c(i,:),x(j,:))/euc_dist(c(k,:),x(j,:)))^(2/(m_exp-1));
        end
        evu(i,j) = 1/sum_d;
    end
end

% defuzzify the membership matrix
for j = 1:m
    if evu(1,j) >= evu(2,j)
        evu(1,j) = 1; evu(2,j) = 0;
    else
        evu(1,j) = 0; evu(2,j) = 1;
    end
end

% Analyze results
ev = EvalSet(:,end)';
rmse(1) = norm(evu(1,:)-ev)/sqrt(length(evu(1,:)));
rmse(2) = norm(evu(2,:)-ev)/sqrt(length(evu(2,:)));
subplot(2,1,1);
if rmse(1) < rmse(2)
    r = 1;
else
    r = 2;
end
str = sprintf('Testing Set RMSE: %f', rmse(r));
disp(str);
ctr = 0;
for i = 1:m
    if evu(r,i)==ev(i)
        ctr = ctr + 1;
    end
end
str = sprintf('Testing Set accuracy: %.2f%%', ctr*100/m);
disp(str);
[m,b,r] = postreg(evu(r,:),ev); % Regression Analysis
disp(sprintf('r = %.3f', r));

