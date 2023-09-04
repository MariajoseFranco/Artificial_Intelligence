% HAC
Z = linkage(train_x);
T = cluster(Z,'maxclust',3);
scatter3(train_x(:,1),train_x(:,2),train_x(:,3),10,T);