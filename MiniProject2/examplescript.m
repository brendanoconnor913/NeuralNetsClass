% EXAMPLE 1, REGRESSION
% Estimate Body Fat Percentage
% * Age (years)
% * Weight (lbs)
% * Height (inches)
% * Neck circumference (cm)
% * Chest circumference (cm)
% * Abdomen 2 circumference (cm)
% * Hip circumference (cm)
% * Thigh circumference (cm)
% * Knee circumference (cm)
% * Ankle circumference (cm)
% * Biceps (extended) circumference (cm)
% * Forearm circumference (cm)
% *Wrist circumference (cm)


[X,T] = bodyfat_dataset;
size(X)
size(T)
net = fitnet(15);
view(net)
[net,tr] = train(net,X,T);
plotperform(tr)
testX = X(:,tr.testInd);
testT = T(:,tr.testInd);
testY = net(testX);
perf = mse(net,testT,testY);
Y = net(X);
plotregression(T,Y)
e = T - Y;
ploterrhist(e)

%%%%%%%%%%%%%%%
% EXAMPLE 2, CLASSIFICATION
% Cancer Detection
[x,t] = ovarian_dataset;
whos
net = patternnet(5);
view(net)

% Also look at net.LW and net.layers{1} and net.layers{2}

% Note that softmax(n) = exp(n)/sum(exp(n))

[net,tr] = train(net,x,t);
testX = x(:,tr.testInd);
testT = t(:,tr.testInd);
testY = net(testX);
testClasses = testY > 0.5;
plotconfusion(testT,testY)


% Class 1 indicate cancer patiencts, class 2 normal patients

figure, plotroc(testT,testY)