

% Global variable initialization to be used for computations
tC      = [ones(1,900) -1*ones(1,2700)];
tT      = [-1*ones(1,900) ones(1,900) -1*ones(1,1800)];
tR      = [-1*ones(1,1800) ones(1,900) -1*ones(1,900)];
tD      = [-1*ones(1,2700) ones(1,900)];
trains  = [1 2; 2 3; 1 3];
tests   = [3; 1; 2];
hall    = zeros(4,4,6);
hindex  = 1; % used for assinging final h's to overall matrix
s1      = [reshape(s1C,2,900) reshape(s1T,2,900) reshape(s1R,2,900) reshape(s1D,2,900)];
s2      = [reshape(s2C,2,900) reshape(s2T,2,900) reshape(s2R,2,900) reshape(s2D,2,900)];
s3      = [reshape(s3C,2,900) reshape(s3T,2,900) reshape(s3R,2,900) reshape(s3D,2,900)];
s4      = [reshape(s4C,2,900) reshape(s4T,2,900) reshape(s4R,2,900) reshape(s4D,2,900)];
s5      = [reshape(s5C,2,900) reshape(s5T,2,900) reshape(s5R,2,900) reshape(s5D,2,900)];
s6      = [reshape(s6C,2,900) reshape(s6T,2,900) reshape(s6R,2,900) reshape(s6D,2,900)];
allS    = {s1 s2 s3 s4 s5 s6};

% Initialize the network
nodes   = 15;
odepth  = 2;
idepth  = 30;
labelSet = con2seq([tC tC tC tC tC; tT tT tT tT tT; tR tR tR tR tR; tD tD tD tD tD]);
net      = narxnet(1:idepth,1:odepth,nodes);
net.layers{2}.transferFcn  = 'tansig';
net.trainFcn               = 'trainbr';
net.trainParam.epochs      = 100;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio   = 0;
net.divideParam.testRatio  = 0;

% 6 Fold validation
for valdsubj=1:6
    output = zeros(4,(3600-(idepth)),3); % Used to store DPM for each iteration
    for itr=1:3
        % Set up training and validation sets
        net = init(net);
        net = openloop(net);
        valdSet                 = con2seq(allS{valdsubj});
        trainingindxs           = 1:6;
        trainingindxs(valdsubj) = [];
        trainingSet             = [];
        for i=trainingindxs
            trainingSet = [trainingSet allS{i}];
        end
        trainingSet = con2seq(trainingSet);

        % Train and save
        [inputs,inputStates,layerStates,targets] = ... 
            preparets(net,trainingSet,{},labelSet);
        net = train(net,inputs,targets,inputStates,layerStates);
        net = closeloop(net);
        save(['SIvaldsub' num2str(valdsubj) 'Itr' num2str(itr) '.mat'], 'net')

        % Predict values for y
        [inputs,inputStates,layerStates,targets] = ...
                    preparets(net,valdSet,{},{});
        yp    = seq2con(sim(net,inputs,inputStates));
        yp = yp{1};
        output(:,:,itr) = yp;
    end
    
    % Calculate subject specific DPM
    htemp = zeros(4,4);
    for i=1:4
        for j=1:4
            averages = zeros(1,3);
            % Get averages for each iteration and then average
            for itr=1:3   
                if(j==1)
                    colintv = 1:(900-idepth);
                else
                    colintv = (900*(j-1)+1-idepth):(900*j)-idepth;
                end
%                 colintv = (900*(j-1)+1):900*j; NOT ADJUSTED FOR INPUT
                meanval = mean(output(i,colintv,itr),2);
                averages(itr) = meanval;
            end
            htemp(i,j) = mean(averages);
        end
    end
    hall(:,:,hindex) = htemp(:,:);
    hindex = hindex + 1;
end

% create overall ROC
[roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(hall);
f = figure;
plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
    title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
    ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
hold on;
saveas(f, 'SubjectIndependentOverall.png')