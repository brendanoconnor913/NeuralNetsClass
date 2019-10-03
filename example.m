[x,t] = simplefit_dataset;
% net = fitnet(10);
% net = train(net,x,t);
% view(net)
% y = net(x);
% plot(t);
% hold on, plot(y);
% % Saving figures
% saveas(plot(y), 'test.png')

nodes = 10;
net = fitnet(nodes);
trainmse = [];
valdmse  = [];

for i=1:5
    net = init(net);
    net = fitnet(nodes);
    net.divideParam.trainRatio = .8;
    net.divideParam.valratio = .2;
    net.divideParam.testratio = 0;
    [net, trainout] = train(net,x,t);
    trainmse(i) = trainout.best_perf;
    valdmse(i) = trainout.best_perf;
    
    y = net(x);
   
    plot(t);
    hold on;
    saveas(plot(y), strcat('test',num2str(i),'.png'))
    cla;
end

trainmean = mean(trainmse);
valdmean  = mean(valdmse);

