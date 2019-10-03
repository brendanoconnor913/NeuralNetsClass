[X,d] = cancer_dataset; %Type help cancer_dataset for more info
X = X(2,:);
w=Xtrain'\dtrain(1,:)'; %Training/MSE linear model creation
y=Xtest'*w; %Activation/testing
[X,Y,T,AUC] = perfcurve(dtest(1,:),y',1);
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC, AUC=' num2str(AUC)])