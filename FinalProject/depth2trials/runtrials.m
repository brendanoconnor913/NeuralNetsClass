
% Various 2 layer AE tests
for l1=[100, 50, 25, 5]
   for l2=[100, 50, 25, 5]
       dprime = genAutoEncoderExperiment([l1,l2],[400,100],[.004,.003],4,[.15,.1]);
       display(['l1:' num2str(l1) ' l2:' num2str(l2) ' dprime ' num2str(dprime)])
   end
end
% dprime = genAutoEncoderExperiment([100,50,5],[400,100,50],[.004,.003, .002],4,[.15,.1,.05]);

% Various 2 layer AE tests
for l1=[100, 50, 25, 5]
   for l2=[100, 50, 25, 5]
       dprime = genAutoEncoderExperiment([l1,l2],[400,100],[.004,.003],4,[.15,.1]);
       display(['l1:' num2str(l1) ' l2:' num2str(l2) ' dprime ' num2str(dprime)])
   end
end

