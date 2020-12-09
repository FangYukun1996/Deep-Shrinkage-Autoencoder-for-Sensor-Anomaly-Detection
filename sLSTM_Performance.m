load('./data/BlockTest.mat');
load('./data/LS_decision.mat');
decision_fc=csvread('./data/decision_fc.txt');
decision_lstm=csvread('./data/decision.txt');

% LLS_AR
count_LLS_AR=0;
TP_LLS_AR=0;
FP_LLS_AR=0;
FN_LLS_AR=0;
for i=1:length(decision_LS)
    if decision_LS(i)==LABEL_ALL(i)
        count_LLS_AR=count_LLS_AR+1;
    end
    if decision_LS(i)==1 && LABEL_ALL(i)==1
        TP_LLS_AR=TP_LLS_AR+1;
    end
    if decision_LS(i)==1 && LABEL_ALL(i)==0
        FP_LLS_AR=FP_LLS_AR+1;
    end
    if decision_LS(i)==0 && LABEL_ALL(i)==1
        FN_LLS_AR=FN_LLS_AR+1;
    end
end
LLS_AR_acc=count_LLS_AR/length(decision_LS);
LLS_AR_precision=TP_LLS_AR/(TP_LLS_AR+FP_LLS_AR);
LLS_AR_recall=TP_LLS_AR/(TP_LLS_AR+FN_LLS_AR);
LLS_AR_f1=2*LLS_AR_precision*LLS_AR_recall/(LLS_AR_precision+LLS_AR_recall);

% FC
count_fc=0;
TP_fc=0;
FP_fc=0;
FN_fc=0;
for i=1:length(decision_fc)
    if decision_fc(i)==LABEL_ALL(i)
        count_fc=count_fc+1;
    end
    if decision_fc(i)==1 && LABEL_ALL(i)==1
        TP_fc=TP_fc+1;
    end
    if decision_fc(i)==1 && LABEL_ALL(i)==0
        FP_fc=FP_fc+1;
    end
    if decision_fc(i)==0 && LABEL_ALL(i)==1
        FN_fc=FN_fc+1;
    end
end
fc_acc=count_fc/length(decision_fc);
fc_precision=TP_fc/(TP_fc+FP_fc);
fc_recall=TP_fc/(TP_fc+FN_fc);
fc_f1=2*fc_precision*fc_recall/(fc_precision+fc_recall);

% s-LSTM
count_lstm=0;
TP_lstm=0;
FP_lstm=0;
FN_lstm=0;
for i=1:length(decision_fc)
    if decision_lstm(i)==LABEL_ALL(i)
        count_lstm=count_lstm+1;
    end
    if decision_lstm(i)==1 && LABEL_ALL(i)==1
        TP_lstm=TP_lstm+1;
    end
    if decision_lstm(i)==1 && LABEL_ALL(i)==0
        FP_lstm=FP_lstm+1;
    end
    if decision_lstm(i)==0 && LABEL_ALL(i)==1
        FN_lstm=FN_lstm+1;
    end
end
lstm_acc=count_lstm/length(decision_lstm);
lstm_precision=TP_lstm/(TP_lstm+FP_lstm);
lstm_recall=TP_lstm/(TP_lstm+FN_lstm);
lstm_f1=2*lstm_precision*lstm_recall/(lstm_precision+lstm_recall);
