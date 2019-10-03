% This function generates autoencoder experiements using the AT&T face
% data set and returns the d' of the test set while also saving the
% images of the training and test ROC's with the corresponding values
% passed for each parameter. All the arguments are lists except SR.

function [dprime]=genAutoEncoderExperiment(laysizes,epochs,L2,SR,SP)

% Get the training and testing data from files
% Create label matricies
trainlabels = zeros(40,240);
testlabels  = zeros(40,160);
class = 1;
for i=1:240
   trainlabels(class,i) = 1;
   if (mod(i, 6) == 0)
       class = class + 1;
   end
end

class = 1;
for i=1:160
    testlabels(class,i) = 1;
    if (mod(i, 4) == 0)
        class = class +1;
    end
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
tmpAE = trainAutoencoder(traindata,laysizes(1), ...
    'MaxEpochs',epochs(1), ...
    'L2WeightRegularization',L2(1), ...
    'SparsityRegularization',SR, ...
    'SparsityProportion',SP(1), ...
    'ScaleData', false);
AEs{1} = tmpAE;

features{end+1} = encode(AEs{1},traindata);

% For network sizes with num layers > 1 repeat
for i=2:size(laysizes,2)
    tmpAE = trainAutoencoder(features{i-1},laysizes(i), ...
        'MaxEpochs',epochs(i), ...
        'L2WeightRegularization',L2(i), ...
        'SparsityRegularization',SR, ...
        'SparsityProportion',SP(i), ...
        'ScaleData', false);
    AEs{end+1} = tmpAE;
    features{i} = encode(tmpAE,features{i-1});
end

patnet = patternnet(40);

% deepnet = stack(AEs{:},patnet);
% view(deepnet)

ytrain = deepnet(traindata);
ytest = deepnet(testdata);

distanceScores = [];
labels = [];
for i=1:size(ytest,2)
    % Compute distance from i'th training image to each training data image
    pairs = ytrain(:,i)';
    for k=1:size(ytrain,2)
        genuine = double(isequal(trainlabels(:,i),trainlabels(:,k)));
        pairs(2,:) = ytrain(:,k)';
        distanceScores(end+1) = -pdist(pairs);
        labels(end+1) = genuine;
    end
end

% Construct filename
baserocstring = ['AE_Depth' num2str(size(laysizes,2))];
baserocstring = [baserocstring '_Sizes'];
for i=1:size(laysizes,2)
   baserocstring = [baserocstring num2str(laysizes(i)) '-'];
end
baserocstring = [baserocstring '_L2Reg'];
for i=1:size(laysizes,2)
   baserocstring = [baserocstring num2str(L2(i)) '-'];
end
baserocstring = [baserocstring '_SparsReg' num2str(SR) '_SparsProps'];
for i=1:size(laysizes,2)
   baserocstring = [baserocstring num2str(SP(i)) '-'];
end
baserocstring = [baserocstring '.png'];

% Plot ROC
[roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(distanceScores, labels);
f = figure;
plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
    title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
    ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
hold on;
saveas(f, ['Train' baserocstring])



% Calculate similar results for test data
distanceScores = [];
labels = [];
for i=1:size(ytest,2)
    % Compute distance from i'th test image to each training data image
    pairs = ytest(:,i)';
    for k=1:size(ytrain,2)
        genuine = double(isequal(testlabels(:,i),trainlabels(:,k)));
        pairs(2,:) = ytrain(:,k)';
        distanceScores(end+1) = -pdist(pairs);
        labels(end+1) = genuine;
    end
    
end


% Plot ROC
[roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(distanceScores, labels);
f = figure;
plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
    title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
    ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
hold on;
saveas(f, ['Test' baserocstring])

% Savenetwork
save(['NET' baserocstring '.mat'], 'deepnet')

dprime = d;
