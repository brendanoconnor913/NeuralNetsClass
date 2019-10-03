

% Global variable initialization to be used for computations
tC = [ones(1,300) -1*ones(1,900)];
tT = [-1*ones(1,300) ones(1,300) -1*ones(1,600)];
tR = [-1*ones(1,600) ones(1,300) -1*ones(1,300)];
tD = [-1*ones(1,900) ones(1,300)];
trains = [1 2; 2 3; 1 3];
tests = [3; 1; 2];
htemp1 = zeros(4,4,3);
htemp2 = zeros(4,4,3);
htemp3 = zeros(4,4,3);
htemp4 = zeros(4,4,3);
htemp5 = zeros(4,4,3);
htemp6 = zeros(4,4,3);
hall  = zeros(4,4,18);
hindex = 1; % used for assinging final h's to overall matrix
hcollection = {htemp1 htemp2 htemp3 htemp4 htemp5 htemp6};
sCcollection = {s1C s2C s3C s4C s5C s6C};
sTcollection = {s1T s2T s3C s4T s5T s6T};
sRcollection = {s1R s2R s3R s4R s5R s6R};
sDcollection = {s1D s2D s3D s4D s5D s6D};

% Initialize the network
nodes = 15;
depth = 30;
sLabels = con2seq([tC tC; tT tT; tR tR; tD tD]);
net = timedelaynet(0:depth,nodes);
net.layers{2}.transferFcn = 'tansig';
net.trainFcn = 'trainbr';
net.trainParam.epochs = 100;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio   = 0;
net.divideParam.testRatio  = 0;
net = init(net);

% Calculate 3-fold results
for itr=1:3
    for subj=1:6
        traini1 = trains(itr,1);
        traini2 = trains(itr,2);
        valdi   = tests(itr);
        sC = sCcollection{subj};
        sT = sTcollection{subj};
        sR = sRcollection{subj};
        sD = sDcollection{subj};

        % Setup training set
        sTrain = con2seq([sC(:,:,traini1) sT(:,:,traini1) sR(:,:,traini1)...
            sD(:,:,traini1) sC(:,:,traini2) sT(:,:,traini2) sR(:,:,traini2)...
            sD(:,:,traini2)]);
        % Train and save net
        [inputs,inputStates,layerStates,targets] = ... 
            preparets(net,sTrain,sLabels);
        net = train(net,inputs,targets,inputStates,layerStates);
        save(['SSsub' num2str(subj) 'Itr' num2str(itr) '.mat'], 'net')
        % Generate validation predictions
        sVald = con2seq([sC(:,:,valdi) sT(:,:,valdi) sR(:,:,valdi) sD(:,:,valdi)]);
        yp = seq2con(sim(net,sVald));
        yp = yp{1};

        % Calculate subject specific DPM
        htemp = hcollection{subj};
        for i=1:4
            for j=1:4
                cols = (300*(j-1)+1):300*j;
                meanval = mean(yp(i,cols),2);
                htemp(i,j,itr) = meanval;
            end
        end
        hcollection{subj} = htemp;
        hall(:,:,hindex) = htemp(:,:,itr);
        hindex = hindex + 1;
    end
end

% Generate ROC's for each person
for subj=1:6
    [roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(hcollection{subj});
    f = figure;
    plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
        title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
        ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
    hold on;
    saveas(f, ['SubjectSpecificSub' num2str(subj) '.png'])
end

% create overall ROC
[roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(hall);
f = figure;
plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
    title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
    ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
hold on;
saveas(f, 'SubjectSpecificOverall.png')
