
% function []=autoencoderexperiment(numlayers, )
% Use your training set to create 
% various auto-encoders by changing size, depth, and learning parameters such as sparsity and 
% regularization. 

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
        disp(fname);
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
        disp(fname);
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

% Set the size of the hidden layer for the autoencoder. make this smaller than
% the input size.
hiddenSize1 = 100;

autoenc1 = trainAutoencoder(traindata,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

feat1 = encode(autoenc1,traindata);

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);
% softnet = trainSoftmaxLayer(feat2,trainlabels,'MaxEpochs',400);
deepnet = stack(autoenc1,autoenc2);
% view(deepnet)

% Perform fine tuning
% deepnet = train(deepnet,traindata,trainlabels);

% ROC for training and testing
ytrain = deepnet(traindata);
% ezroc3(ytrain, trainlabels);

ytest = deepnet(testdata);
% ezroc3(ytest, testlabels)


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
saveas(f, ['AE_Depth2_Size100.50_L2.004_SparsReg4_SparsProp.1.png'])