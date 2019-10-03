load('P.mat')
load('T.mat')

REGSPREAD  = 100;
REGGOAL    = .00000000000000000000000000000000000000000000001;
VALDGOAL   = .1;
mseVald    = 10000000;
nodes      = 1;
stepsize   = 50;
[trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,.6,.2,.2);
[trainT,valT,testT] = divideind(T,trainInd,valInd,testInd);
maxnodes   = size(trainP,2);
allmse     = [];

while (REGSPREAD > 0)
    display(strcat("regspread: ",num2str(REGSPREAD)))
    nodes = 1;
    while (mseVald > VALDGOAL && nodes <= maxnodes)
        prevVald = mseVald;
        regnet   = newrb(trainP,trainT,REGGOAL,REGSPREAD, nodes, nodes);
        yRegVald = sim(regnet, valP);
        mseVald = mse(regnet, yRegVald, valT);
        display(strcat("valdmse: ",num2str(mseVald)));
        allmse(REGSPREAD, nodes) = mseVald;
        nodes = nodes + stepsize;
    end
    REGSPREAD = REGSPREAD - 25;
end

% Find minimum vald error configuration
allmse(allmse == 0) = 999;
[ospread,onodes] = find(allmse==min(min(allmse)));
display(strcat("optimal spread:",num2str(ospread)," optimal nodes:",num2str(onodes)));
display(strcat("min vald error: ",num2str(allmse(ospread,onodes))));
% Use optimal settings to calculate results
regnet = newrb(trainP,trainT,REGGOAL,ospread,onodes,onodes);

% Calculate training results
yRegTrain = sim(regnet,trainP);
mseRegTrain = mse(regnet, yRegTrain, trainT);
display(strcat("train:",num2str(mseRegTrain)));
trainT = trainT == 1;
% % Plot ROC
h = figure('visible','off');
hold on;
plotroc(trainT, yRegTrain)
saveas(h, strcat('valStoptrainROC.png'))

% Calculate validation results
yRegVald = sim(regnet,valP);
mseRegVald = mse(regnet, yRegVald, valT);
display(strcat("vald:",num2str(mseRegVald)));
valT = valT == 1;
% % Plot ROC
h = figure('visible','off');
hold on;
plotroc(valT, yRegVald)
saveas(h, strcat('valStopvalidationROC.png'))

% Calculate test results
yRegTest = sim(regnet,testP);
mseRegTest = mse(regnet, yRegTest, testT);
display(strcat("test:",num2str(mseRegTest)));
testT = testT == 1;
% % Plot ROC
h = figure('visible','off');
hold on;
plotroc(testT, yRegTest)
saveas(h, strcat('valStoptestROC.png'))
