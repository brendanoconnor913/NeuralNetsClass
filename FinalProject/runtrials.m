
% Various 2 layer AE tests
% for l1=[100, 50, 25, 5]
%    for l2=[100, 50, 25, 5]
%        dprime = genAutoEncoderExperiment([l1,l2],[400,100],[.004,.003],4,[.15,.1]);
%        display(['l1:' num2str(l1) ' l2:' num2str(l2) ' dprime ' num2str(dprime)])
%    end
% end

% for l1=[400, 300, 200]
%    for l2=[100, 50, 25]
%        dprime = genAutoEncoderExperiment([l1,l2],[400,100],[.004,.003],4,[.15,.1]);
%        display(['l1:' num2str(l1) ' l2:' num2str(l2) ' dprime ' num2str(dprime)])
%    end
% end

% Various 3 layer AE tests
% l1 = 100;
% for l2=[100, 50, 25]
%    for l3=[25, 10, 5]
%         dprime = genAutoEncoderExperiment([l1,l2,l3],[400,100,50],[.004,.003,.002],4,[.15,.1,.05]);
%         display(['l1:' num2str(l1) ' l2:' num2str(l2) ' dprime ' num2str(dprime)])
%    end
% end

% l1 = 400;
% l2 = 100;
% for layer1l2=[.006, .004, .002]
%    for layer2l2=[.003, .002, .001]
%         dprime = genAutoEncoderExperiment([l1,l2],[400,100],[layer1l2,layer2l2],4,[.15,.1]);
%         display(['l1:' num2str(layer1l2) ' l2:' num2str(layer2l2) ' dprime ' num2str(dprime)])
%    end
% end

% l1 = 400;
% l2 = 100;
% layer1l2 = .004;
% layer2l2 = .003;
% layer1sp = .3;
% layer2sp = .5;
% 
% dprime = genAutoEncoderExperiment([l1,l2],[400,100],[layer1l2,layer2l2],4,[layer1sp,layer2sp]);
% display(['l1:' num2str(layer1sp) ' l2:' num2str(layer2sp) ' dprime ' num2str(dprime)])


