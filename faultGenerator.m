%% ԭʼ���ݼ���
data=csvread('./data/data5_EMD_DWT.txt');
X=data(:,8:13);%��ȡ�ߵ�����
Y=data(:,1:2);

%% �������ͷ���
% m=size(X);%ȡ������

% %Strong Noise
% X_noise=zeros(size(X));
% for i=1:m(2)%���ÿһ������
%     mu=0.01*mean(X(:,i));
%     sigma=0.05*var(X(:,i));
%     X_noise(:,i)=X(:,i)+mu+sigma*randn(size(X(:,i)));
% end

% abrupt impulse
% X_abrupt=X;
% X_abrupt_label=ones(length(X_abrupt),1);
% stride=zeros(2,m(2));%��һ�м�¼��ʼ���ڶ��м�¼����
% for i=1:m(2)%���ÿһ������
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
% drift=zeros(2,m(2));%��һ�м�¼Ư�Ƶ㣬�ڶ��м�¼Ư�Ʒ���
% for i=1:m(2)%���ÿһ������
%     biasTime=randi([3000,length(X)-3000],1,1);
%     biasAmp=3*(-1+2*rand(1,1))*mean(X(:,i));% -1+2*rand(1,1)���ϵ��ȡֵ��Χ[-1,1]
%     drift(:,i)=[biasTime;biasAmp];
%     X_bias(biasTime:length(X),i)=X_bias(biasTime:length(X),i)+biasAmp;
% end
% 
% % Block
% X_block=X;
% breakTime=zeros(1,m(2));%��¼������ʱ���
% for i=1:m(2)%���ÿһ������
%     breakdownTime=randi([3000,length(X)-3000],1,1);
%     breakTime(i)=breakdownTime;
%     X_block(breakdownTime:length(X),i)=X_block(breakdownTime,i);
% end

Y_block=Y;
m=size(Y);
TestDataSize=1000;
BlockAnomalyTestData=zeros(length(Y),TestDataSize);
BlockTime=zeros(TestDataSize,1);%��¼������ʱ���
FLAG=zeros(TestDataSize,1);%��¼�ĸ�����������block
for i=1:TestDataSize
    flag=randi(m(2));%���ȷ���ĸ������ᷢ��block
    FLAG(i)=flag;
    breakdownTime=randi([3000,length(Y)-3000],1,1);
    BlockTime(i)=breakdownTime;
    Y_block(breakdownTime:length(Y),flag)=Y_block(breakdownTime,flag);
    BlockAnomalyTestData(:,i)=Y_block(:,flag);
    Y_block=Y;%��ԭ���Ա���һ�ε���������
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
% MissTime=zeros(1,m(2));%��¼�ϵ��ʱ���
% for i=1:m(2)%���ÿһ������������ÿһ���������Բ�ͬ������������һ���綼����
%     miss_time=randi([3000,length(X)-3000],1,1);
%     MissTime(i)= miss_time;
%     X_miss(miss_time:length(X),i)=0;
% end

