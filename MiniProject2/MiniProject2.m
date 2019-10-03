load('P.mat')
load('T.mat')

REGSPREAD  = 100;
REGGOAL    = .7;
mseRegVald = 0;

while(mseRegVald < 0.47 || mseRegVald > .53)
    display(REGSPREAD)
    [trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,.6,.2,.2);
    [trainT,valT,testT] = divideind(T,trainInd,valInd,testInd);

    % Train net
    regnet   = newrb(trainP,trainT,REGGOAL,REGSPREAD);

    % Calculate training results
    yRegTrain = sim(regnet,trainP);
    mseRegTrain = mse(regnet, yRegTrain, trainT);
    display(strcat("train:",num2str(mseRegTrain)));
    trainT = trainT == 1;
    % % Plot ROC
    h = figure('visible','off');
    hold on;
    plotroc(trainT, yRegTrain)
    saveas(h, strcat('regtrainROC.png'))
    % % Plot confusion matrix
    regpred = yRegTrain > 0;
    hold on;
    plotconfusion(trainT,regpred);
    saveas(h, strcat('regtrainConfAt0.png'))

    % Calculate validation results
    yRegVald = sim(regnet,valP);
    mseRegVald = mse(regnet, yRegVald, valT);
    display(strcat("vald:",num2str(mseRegVald)));
    valT = valT == 1;
    % % Plot ROC
    h = figure('visible','off');
    hold on;
    plotroc(valT, yRegVald)
    saveas(h, strcat('regvalidationROC.png'))
    % % Plot confusion matrix
    regpred = yRegVald > 0;
    hold on;
    plotconfusion(valT,regpred);
    saveas(h, strcat('regvalidationConfAt0.png'))

    % Calculate test results
    yRegTest = sim(regnet,testP);
    mseRegTest = mse(regnet, yRegTest, testT);
    display(strcat("test:",num2str(mseRegTest)));
    testT = testT == 1;
    % % Plot ROC
    h = figure('visible','off');
    hold on;
    plotroc(testT, yRegTest)
    saveas(h, strcat('regtestROC.png'))
    % % Plot confusion matrix
    regpred = yRegTest > 0;
    hold on;
    plotconfusion(testT,regpred);
    saveas(h, strcat('regtestConfAt0.png'))
    
    if (mseRegVald - .5 > 0)
        REGSPREAD = REGSPREAD - 25;
    else
        REGSPREAD = REGSPREAD + 25;
    end
end

% % Exact net
EXACTSPREAD  = 100;
mseExactVald = 0;
while(mseExactVald < 0.47 || mseExactVald > .53)
    display(EXACTSPREAD)
    [trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,.6,.2,.2);
    [trainT,valT,testT] = divideind(T,trainInd,valInd,testInd);

    % Train net
    exactnet = newrbe(trainP,trainT,EXACTSPREAD);

    % Calculate training results
    yExactTrain = sim(exactnet,trainP);
    mseExactTrain = mse(regnet, yExactTrain, trainT);
    display(strcat("train:",num2str(mseExactTrain)));
    trainT = trainT == 1;
    % % Plot ROC
    h = figure('visible','off');
    hold on;
    plotroc(trainT, yExactTrain)
    saveas(h, strcat('exacttrainROC.png'))
    % % Plot confusion matrix
    exactpred = yExactTrain > 0;
    hold on;
    plotconfusion(trainT,exactpred);
    saveas(h, strcat('exacttrainConfAt0.png'))

    % Calculate validation results
    yExactVald = sim(exactnet,valP);
    mseExactVald = mse(exactnet, yExactVald, valT);
    display(strcat("vald:",num2str(mseExactVald)));
    valT = valT == 1;
    % % Plot ROC
    h = figure('visible','off');
    hold on;
    plotroc(valT, yExactVald)
    saveas(h, strcat('exactvalidationROC.png'))
    % % Plot confusion matrix
    exactpred = yExactVald > 0;
    hold on;
    plotconfusion(valT,exactpred);
    saveas(h, strcat('exactvalidationConfAt0.png'))

    % Calculate test results
    yExactTest = sim(exactnet,testP);
    mseExactTest = mse(exactnet, yExactTest, testT);
    display(strcat("test:",num2str(mseExactTest)));
    testT = testT == 1;
    % % Plot ROC
    h = figure('visible','off');
    hold on;
    plotroc(testT, yExactTest)
    saveas(h, strcat('exacttestROC.png'))
    % % Plot confusion matrix
    exactpred = yExactTest > 0;
    hold on;
    plotconfusion(testT,exactpred);
    saveas(h, strcat('exacttestConfAt0.png'))
    
    if (mseExactVald - .5 > 0)
        EXACTSPREAD = EXACTSPREAD - 15;
    else
        EXACTSPREAD = EXACTSPREAD + 25;
    end
end
