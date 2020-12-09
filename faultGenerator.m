%% 原始数据加载
data=csvread('./data/data5_EMD_DWT.txt');
X=data(:,8:13);%仅取惯导数据
Y=data(:,1:2);

%% 错误类型分类
% m=size(X);%取特征数

% %Strong Noise
% X_noise=zeros(size(X));
% for i=1:m(2)%针对每一个特征
%     mu=0.01*mean(X(:,i));
%     sigma=0.05*var(X(:,i));
%     X_noise(:,i)=X(:,i)+mu+sigma*randn(size(X(:,i)));
% end

% abrupt impulse
% X_abrupt=X;
% X_abrupt_label=ones(length(X_abrupt),1);
% stride=zeros(2,m(2));%第一行记录起始，第二行记录步长
% for i=1:m(2)%针对每一个特征
%     start=randi([0,400],1,1);
%     span=randi([100,400],1,1);
%     stride(:,i)=[start;span];
%     for j=start:span:length(X)
%         X_abrupt(j,i)=X(j,i)+10*mean(X(:,i));
%     end
%     X_abrupt_seq=stride(1,i):stride(2,i):length(X_abrupt);
%     X_abrupt_label(X_abrupt_seq)=-1;
% end
% save('abrupt.mat','X_abrupt','X_abrupt_label');

% % drift
% X_bias=X;
% drift=zeros(2,m(2));%第一行记录漂移点，第二行记录漂移幅度
% for i=1:m(2)%针对每一个特征
%     biasTime=randi([3000,length(X)-3000],1,1);
%     biasAmp=3*(-1+2*rand(1,1))*mean(X(:,i));% -1+2*rand(1,1)这个系数取值范围[-1,1]
%     drift(:,i)=[biasTime;biasAmp];
%     X_bias(biasTime:length(X),i)=X_bias(biasTime:length(X),i)+biasAmp;
% end
% 
% % Block
% X_block=X;
% breakTime=zeros(1,m(2));%记录卡死的时间点
% for i=1:m(2)%针对每一个特征
%     breakdownTime=randi([3000,length(X)-3000],1,1);
%     breakTime(i)=breakdownTime;
%     X_block(breakdownTime:length(X),i)=X_block(breakdownTime,i);
% end

Y_block=Y;
m=size(Y);
TestDataSize=1000;
BlockAnomalyTestData=zeros(length(Y),TestDataSize);
BlockTime=zeros(TestDataSize,1);%记录卡死的时间点
FLAG=zeros(TestDataSize,1);%记录哪个特征发生了block
for i=1:TestDataSize
    flag=randi(m(2));%随机确定哪个特征会发生block
    FLAG(i)=flag;
    breakdownTime=randi([3000,length(Y)-3000],1,1);
    BlockTime(i)=breakdownTime;
    Y_block(breakdownTime:length(Y),flag)=Y_block(breakdownTime,flag);
    BlockAnomalyTestData(:,i)=Y_block(:,flag);
    Y_block=Y;%还原，以便下一次的数据生成
end
figure;
k=randi(TestDataSize);
flag=FLAG(k);
plot(Y(:,flag));hold on;
plot(BlockAnomalyTestData(:,k));
save('block.mat','BlockAnomalyTestData','FLAG');

% 
% % Miss
% X_miss=X;
% MissTime=zeros(1,m(2));%记录断点的时间点
% for i=1:m(2)%针对每一个特征。假设每一个特征来自不同传感器。否则一掉电都掉电
%     miss_time=randi([3000,length(X)-3000],1,1);
%     MissTime(i)= miss_time;
%     X_miss(miss_time:length(X),i)=0;
% end

