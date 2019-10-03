% d1 = [s2C(:,:,1) s2T(:,:,1) s2R(:,:,1) s2D(:,:,1)];
% d2 = [s2C(:,:,2) s2T(:,:,2) s2R(:,:,2) s2D(:,:,2)];
% 
% o = d1/d2;
% out = o*d2;
% noise = normrnd(5,0,2,1200);
% d1n = d1 + noise;

% Global variable initialization to be used for computations
% tC = [ones(1,300) -1*ones(1,900)];
% tT = [-1*ones(1,300) ones(1,300) -1*ones(1,600)];
% tR = [-1*ones(1,600) ones(1,300) -1*ones(1,300)];
% tD = [-1*ones(1,900) ones(1,300)];
% trains = [1 2; 2 3; 1 3];
% tests = [3; 1; 2];
% htemp = zeros(4,4,3);
% hall  = zeros(4,4,18);
% hindex = 1; % used for assinging final h's to overall matrix
% nodes = 15;
% depth = 45;
% sLabels = con2seq([tC tC; tT tT; tR tR; tD tD]);
% net = timedelaynet(0:depth,nodes);
% net.layers{2}.transferFcn = 'tansig';
% net.trainFcn = 'trainbr';
% net.trainParam.epochs = 100;
% net.divideParam.trainRatio = 1;
% net.divideParam.valRatio   = 0;
% net.divideParam.testRatio  = 0;
% 
% % Calculate 3-fold results
% for itr=1:3
%     traini1 = trains(itr,1);
%     traini2 = trains(itr,2);
%     valdi   = tests(itr);
%     
%     s1train = con2seq([s1C(:,:,traini1) s1T(:,:,traini1) s1R(:,:,traini1)...
%         s1D(:,:,traini1) s1C(:,:,traini2) s1T(:,:,traini2) s1R(:,:,traini2)...
%         s1D(:,:,traini2)]);
%     [inputs,inputStates,layerStates,targets] = ... 
%         preparets(net,s1train,sLabels);
%     net = train(net,inputs,targets,inputStates,layerStates);
%     save(['SSsub1Itr' num2str(itr) '.mat'], 'net')
%     
%     s1vald = con2seq([s1C(:,:,valdi) s1T(:,:,valdi) s1R(:,:,valdi) s1D(:,:,valdi)]);
%     yp = seq2con(sim(net,s1vald));
%     yp = yp{1};
%     
% %     Calculate subject specific DPM
%     for i=1:4
%         for j=1:4
%             cols = (300*(j-1)+1):300*j;
%             val = mean(yp(i,cols),2);
%             htemp(i,j,itr) = val;
%         end
%     end
%     hall(:,:,hindex) = htemp(:,:,itr);
%     hindex = hindex + 1;
% end
% 
% % create the dpm for each iteration then use that to create roc
% [roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(htemp);
% f = figure;
% plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
%     title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
%     ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
% hold on;
% saveas(f, ['SubjectSpecificSub' num2str(1) '.png'])

for k = test11
    display(k{1})
end
