
laysizes1 = [400 100];
epochs1   = [400 100];
L21       = [.004 .003];
SR       = 4;
SP1       = [.3 .5];

% laysizes2 = [400 100];
% epochs2   = [400 100];
% L22       = [.004 .003];
% SP2       = [.3 .5];

traindistanceScores = [];
finaltrainlabels    = [];
testdistanceScores  = [];
finaltestlabels     = [];

for class=1:40
    % Get the training and testing data from files
    % Create label matricies
    trainlabels = zeros(1,240);
    testlabels  = zeros(1,160);

    for i=1:240
       trainlabels(i) = ceil(i/6) == class;
    end

    for i=1:160
       testlabels(i) = ceil(i/4) == class;
    end

    ddir = dir('./orl_faces');
    traindata = []; % Matrix to store all of our images
    inc = 1;
    % Read through all of the image directories and get the first half of each
    % directory (6 of the 10 images)
    for k = 3:length(ddir) % discard '.' and '..'
        if (ddir(k).isdir)
            fname = strcat('orl_faces/',ddir(k).name);
    %         disp(fname);
            imds = imageDatastore(fname);
            % load all images into matrix
            for i = 1:6
                % Scale the image and add to our matrix of all images
                m = imresize(double(readimage(imds,i)), .2);
                l = reshape(m, [437,1]);
                traindata(:,inc) = l;
                inc = inc + 1;
            end
        end
    end

    % Load appropriate test data into matrix
    testdata = [];
    inc = 1;
    for k = 3:length(ddir) % discard '.' and '..'
        if (ddir(k).isdir)
            fname = strcat('orl_faces/',ddir(k).name);
    %         disp(fname);
            imds = imageDatastore(fname);
            % load all test images into matrix
            for i = 7:10
                m = imresize(double(readimage(imds,i)), .2);
                l = reshape(m, [437,1]);
                testdata(:,inc) = l;
                inc = inc + 1;
            end
        end
    end

    % Use training set to create AEs for testing
    AEs = {};
    features = {};
    tmpAE = trainAutoencoder(traindata,laysizes1(1), ...
        'MaxEpochs',epochs1(1), ...
        'L2WeightRegularization',L21(1), ...
        'SparsityRegularization',SR, ...
        'SparsityProportion',SP1(1), ...
        'ScaleData', false);
    AEs{1} = tmpAE;
    features{end+1} = encode(AEs{1},traindata);
    tmpAE = trainAutoencoder(features{1},laysizes1(2), ...
        'MaxEpochs',epochs1(2), ...
        'L2WeightRegularization',L21(2), ...
        'SparsityRegularization',SR, ...
        'SparsityProportion',SP1(2), ...
        'ScaleData', false);
    AEs{2} = tmpAE;
    features{2} = encode(tmpAE,features{1});
    patnet = patternnet(100);
    patnet = train(patnet,features{2},trainlabels);
    deepnet1 = stack(AEs{:},patnet);
    % view(deepnet)
    
    % Use training set to create AEs for testing
    AEs = {};
    features = {};
    tmpAE = trainAutoencoder(traindata,laysizes1(1), ...
        'MaxEpochs',epochs1(1), ...
        'L2WeightRegularization',L21(1), ...
        'SparsityRegularization',SR, ...
        'SparsityProportion',SP1(1), ...
        'ScaleData', false);
    AEs{1} = tmpAE;
    features{end+1} = encode(AEs{1},traindata);
    tmpAE = trainAutoencoder(features{1},laysizes1(2), ...
        'MaxEpochs',epochs1(2), ...
        'L2WeightRegularization',L21(2), ...
        'SparsityRegularization',SR, ...
        'SparsityProportion',SP1(2), ...
        'ScaleData', false);
    AEs{2} = tmpAE;
    features{2} = encode(tmpAE,features{1});
    patnet = patternnet(100);
    patnet = train(patnet,features{2},trainlabels);
    deepnet2 = stack(AEs{:},patnet);

    ytrain1     = deepnet1(traindata);
    ytest1      = deepnet1(testdata);
    ytrain2     = deepnet2(traindata);
    ytest2      = deepnet2(testdata);
    ytrainfinal = (ytrain1 + ytrain2) / 2;
    ytestfinal  = (ytest1 + ytest2) / 2;

    traindistanceScores = [traindistanceScores ytrainfinal];
    finaltrainlabels    = [finaltrainlabels trainlabels];
    testdistanceScores  = [testdistanceScores ytestfinal];
    finaltestlabels     = [finaltestlabels testlabels];
end

% Construct filename
baserocstring = ['AE_Depth' num2str(size(laysizes1,2))];
baserocstring = [baserocstring '_Sizes'];
for i=1:size(laysizes1,2)
   baserocstring = [baserocstring num2str(laysizes1(i)) '-'];
end
baserocstring = [baserocstring '_L2Reg'];
for i=1:size(laysizes1,2)
   baserocstring = [baserocstring num2str(L21(i)) '-'];
end
baserocstring = [baserocstring '_SparsReg' num2str(SR) '_SparsProps'];
for i=1:size(laysizes1,2)
   baserocstring = [baserocstring num2str(SP1(i)) '-'];
end
baserocstring = [baserocstring '.png'];

% Plot ROC
[roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(traindistanceScores, finaltrainlabels);
f = figure;
plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
    title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
    ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
hold on;
saveas(f, ['Train' baserocstring])

% Plot ROC
[roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(testdistanceScores, finaltestlabels);
f = figure;
plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
    title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
    ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
hold on;
saveas(f, ['Test' baserocstring])

% Savenetwork
save(['NET1' baserocstring '.mat'], 'deepnet1')
save(['NET2' baserocstring '.mat'], 'deepnet2')

dprime = d;