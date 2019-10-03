% MATLAB 2D and 3D ROC demo

[X,Y,T,AUC] = perfcurve([ones(1,100),zeros(1,100)],0.4*randn(1,200)+[ones(1,100),zeros(1,100)],1);

plot(X,Y)

xlabel('False positive rate')

ylabel('True positive rate')

title(['2D ROC Example, AUC=' num2str(AUC)])

[X,Y,T,AUC] = perfcurve([ones(1,100),zeros(1,100)],0.4*randn(1,200)+[ones(1,100),zeros(1,100)],1);

figure,plot3(X,Y,T)

xlabel('False positive rate')

ylabel('True positive rate')

title(['3D ROC Example, AUC=' num2str(AUC)])

%Note: MATLAB version seems not to be able to do multiple classes and EER

% This is an older and less useful version that directly plts the ROC: plotroc(targets,outputs)

% Confusion matric example

 
% 
D = [1 1 1 1 0 0 0 0];      % Desired (ground truth class labels)

Y = [1 1 1 0 0 0 1 1];  % Predicted class labels

 

[Cmat,order] = confusionmat(D,Y)

plotconfusion(D,Y)  %Grapohical (not for multi-dim)

 

D = [1 1 2 2 3 3];      % Desired (ground truth class labels)

Y = [1 1 2 3 4 NaN];    % Predicted class labels

 

[Cmat,order] = confusionmat(D,Y)

 

%Linear regression: w = X\d for Xw=d, X being N*dim(w) input data and d

%corresponding desired vals

[X,d] = cancer_dataset; %Type help cancer_dataset for more info

 

w=X'\d(2,:)'; %Training/MSE model creation

y=X'*w; %Activation/testing (we'll talk about big no no here later)

[Cmat,order] = confusionmat(d(2,:),double(y>0.5))

plotroc(d(2,:),y')

%NN quick example

net = patternnet(10);

net = train(net,X,d(2,:));

y2 = net(X);

figure,plotroc(d(2,:),y2);