load('./data/BlockTest.mat');
load('./data/LS_max_err');
m=size(TEST_DATA_ALL);
decision_LS=zeros(m(2),1);
X_predict=zeros(m);
%设置滑窗大小
W=300;%滑窗要根据研究对象定大小。这里，100是1秒的采样数
factors=4.452528;
for i=1:10
    for j=W+1:m(1)
        X_predict(:,i)=[W+1,1]*([(1:W)',ones(W,1)]\TEST_DATA_ALL(j-W:j-1,i));%线性回归模型y=ax+b短时预测。
    end
    err=sqrt((X_predict(W+1:end,i)-TEST_DATA_ALL(W+1:end,i)).^2);
    err_m=err_max(FLAG_ALL(i));
%     beta=factors./(1+exp(err))+1;
    beta=1;
    delta=beta*err_m;
    phi=sign(delta-err);
    if sum(phi)<m(1)-W
        decision_LS(i)=1;
    end
end
save('LS_decision.mat','decision_LS');
