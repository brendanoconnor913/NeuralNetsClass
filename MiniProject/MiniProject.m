[X,L] = bodyfat_dataset;

% Save last 50 data samples for testing
xtrain = X(:,1:202);
ltrain = L(:, 1:202);

xtest = X(:, 202:252);
ltest = L(:, 202:252);

net = fitnet(1);
% Train at reg params = 0 .1 .5 for train ratio = .4
for regparam = [0, .1, .5]
    for trainratio = [.8, .4]
        valdratio = 1 - trainratio;

        if(trainratio == .8 && regparam ~= 0)
            continue
        end
        
        % Train network at various size of nodes
        for nodes = [10, 2, 50]
            net = fitnet(nodes);
            trainmse = [];
            valdmse  = [];
            testmse  = [];

            % Train each config 10 times and save the mse's for train,
            % vald, test
            for i=1:10
                net = init(net);
                net = fitnet(nodes);
                net.performParam.regularization = regparam;
                net.divideParam.trainRatio = trainratio;
                net.divideParam.valratio = valdratio;
                net.divideParam.testratio = 0;
                [net, trainout] = train(net,xtrain,ltrain);
                trainmse(i,1) = trainout.best_perf;
                valdmse(i,1) = trainout.best_vperf;
                y = net(xtest);
                testmse(i,1) = mse(net,ltest,y);
            end
            
            % Calculate mean and variance for each of the mse's
            trainmean = mean(trainmse);
            trainvar  = var(trainmse);
            valdmean  = mean(valdmse);
            valdvar   = var(valdmse);
            testmean  = mean(testmse);
            testvar   = var(testmse);
            
            % Output to console the corresponding values for each config
            display(strcat('nodes',num2str(nodes),' tr', ... 
                num2str(trainratio),' reg', num2str(regparam)))
            display(trainmean)
            display(trainvar)
            display(valdmean)
            display(valdvar)
            display(testmean)
            display(testvar)

            % Plot results
            h = figure;
            hold on;
            p1 = plot(trainmse); l1 = 'trainmse';
            p2 = plot(valdmse); l2 = 'valdmse';
            p3 = plot(testmse); l3 = 'testmse';
            legend(l1,l2,l3);
            saveas(h, strcat('performance.nodes',num2str(nodes),'.tr', ... 
                num2str(trainratio),'.reg', num2str(regparam), ...
                '.png'))
        end
    end
end

