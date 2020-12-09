data=csvread('./data/p2_log_20190524_outlierExc_winLS.txt');
X_measure=data(:,1:2);

%设置滑窗大小
W=300;%滑窗要根据研究对象定大小。这里，100是1秒的采样数

m=size(X_measure);%这里主要是要提取向量的维数，即多少个量刻画了研究对象的状态
X_predict=zeros(size(X_measure));X_predict(1:W,:)=X_measure(1:W,:);
X_cor=zeros(size(X_measure));X_cor(1:W,:)=X_measure(1:W,:);
e_r_up=0.2;%给定相对误差限上限
for i=W+1:length(X_measure)
    for j=1:m(2)
        X_predict(i,j)=[W+1,1]*([(1:W)',ones(W,1)]\X_measure(i-W:i-1,j));%线性回归模型y=ax+b短时预测。不建议使用超过二阶的多项式模型
        %给定预测值和测量值所占权重的规则
        %这个阈值的确定还是有待商榷，主要是减去baseline之后极有可能出现0值
        if abs((X_measure(i,j)-X_predict(i,j))/X_predict(i,j))>e_r_up %相对误差超过e_r，认为测量值不可信，权重置为0
            theta=0;
        else
            %这个阈值的确定还是有待商榷，主要是减去baseline之后极有可能出现0值
            theta=(1-abs((X_measure(i,j)-X_predict(i,j))/X_predict(i,j))-(1-e_r_up))*10;%[(1-er)-(1-er_up)]*10,把范围映射到[0,1],er越小占权越大
        end
        X_cor(i,j)=theta*X_measure(i,j)+(1-theta)*X_predict(i,j);%滑动平均
    end
end
err_max_lat=max(sqrt((X_predict(W+1:end,1)-X_measure(W+1:end,1)).^2));
err_max_lon=max(sqrt((X_predict(W+1:end,2)-X_measure(W+1:end,2)).^2));
err_max=[err_max_lat,err_max_lon];
save('LS_max_err.mat','err_max');
% figure;
% plot( X_predict(:,1));
% figure;
% plot( X_predict(:,2));