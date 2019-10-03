
% "Prefold" section
for nodes=[8 15]
    for idepth=[15 30]
        for odepth=[2 5]
            h = zeros(4,4,5);
            % 5 iterations for each configuration
            for itr=1:5
                % Set up training set
                s2train = con2seq([s2C(:,:,1) s2T(:,:,1) s2R(:,:,1) s2D(:,:,1)]);
                tC = [ones(1,300) -1*ones(1,900)];
                tT = [-1*ones(1,300) ones(1,300) -1*ones(1,600)];
                tR = [-1*ones(1,600) ones(1,300) -1*ones(1,300)];
                tD = [-1*ones(1,900) ones(1,300)];
                s2Labels = con2seq([tC; tT; tR; tD]);
                
                % Set up validation set
                s2vald  = con2seq([s2C(:,:,2) s2T(:,:,2) s2R(:,:,2) s2D(:,:,2)]);
                net = narxnet(1:idepth,1:odepth,nodes);
                net = init(net);
                net = openloop(net);
                net.layers{2}.transferFcn = 'tansig';
                net.trainFcn = 'trainbr';
                net.trainParam.epochs = 100;
                net.divideParam.trainRatio = 1;
                net.divideParam.valRatio   = 0;
                net.divideParam.testRatio  = 0;

                % Train net and save
                [inputs,inputStates,layerStates,targets] = ... 
                    preparets(net,s2train,{},s2Labels);
                net = train(net,inputs,targets,inputStates,layerStates);
                net = closeloop(net);
                save(['nnIODepth' num2str(idepth) '_' num2str(odepth) 'Nodes' num2str(nodes) '.mat'], 'net')
%                 view(net)
                
                % Test net
                [inputs,inputStates,layerStates,targets] = ...
                    preparets(net,s2vald,{},{});
                yp = seq2con(sim(net,inputs,inputStates));
                yp = yp{1};
                % Calculate the DPM
                for i=1:4
                    for j=1:4
                        if(j==1)
                            cols = 1:(300-idepth);
                        else
                            cols = (300*(j-1)+1-idepth):(300*j)-idepth;
                        end
                        val = mean(yp(i,cols),2);
                        h(i,j,itr) = val;
                    end
                end
            end

            % Plot and save ROC
            [roc,EER,area,EERthr,ALLthr,d,gen,imp,rbst] = ezroc3(h);
            f = figure;
            plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),...
                title(['ROC Curve;   EER=' num2str(EER) ',   Area=' num2str(area) ...
                ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
            hold on;
            saveas(f, ['prefoldEvalRocIODepth' num2str(idepth) '_' num2str(odepth) 'Nodes' num2str(nodes) '.png'])
        end
    end
end

