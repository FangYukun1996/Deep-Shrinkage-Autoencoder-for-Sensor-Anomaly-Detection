%% 原始数据加载
load('./data/data4vAv.mat');
X=X(5000:19999,1);

%% 错误类型分类
% Instant
X_instant=X;
short_point=randi([1000,3000],1,5);short_point=cumsum(short_point);
X_instant(short_point)=X(short_point)+mean(X)*randn(size(X(short_point)));

figure;
plot(X_instant,'r','LineWidth',3);hold on;
plot(X,'k','LineWidth',3);hold on
title('Instant','FontSize',24);
set(gca,'FontSize',20);

% Constant
X_constant=X;
breakdownTime=randi([3000,length(X)-3000],1,1);%卡死的时间点
X_constant(breakdownTime:length(X))=X(breakdownTime);

figure;
plot(X,'k','LineWidth',3);hold on
plot(X_constant,'r','LineWidth',3);hold on;
title('Constant','FontSize',24);
set(gca,'FontSize',20);

% Bias
X_bias=X;
biasTime=randi([3000,length(X)-3000],1,1);
biasAmp=3*(-1+2*rand(1,1))*mean(X);% -1+2*rand(1,1)这个系数取值范围[-1,1]
X_bias(biasTime:length(X))=X_bias(biasTime:length(X))+biasAmp;

figure;
plot(X,'k','LineWidth',3);hold on
plot(X_bias,'r','LineWidth',3);hold on;
title('Bias','FontSize',24);
set(gca,'FontSize',20);

% Gradual Drift
X_drift=X;
driftTime=randi([3000,length(X)-3000],1,1);
for i=driftTime:length(X)
    X_drift(i)=X(i)-0.0005*(i-driftTime+1);
end

figure;
plot(X,'k','LineWidth',3);hold on
plot(X_drift,'r','LineWidth',3);hold on;
title('Gradual Drift','FontSize',24);
set(gca,'FontSize',20);   

% Miss
X_miss=X;
missTime_time=randi([3000,length(X)-3000],1,1);
X_miss(missTime_time:length(X))=0;

figure;
plot(X,'k','LineWidth',3);hold on
plot(X_miss,'r','LineWidth',3);hold on;
title('Miss','FontSize',24);
set(gca,'FontSize',20);
