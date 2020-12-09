cD=0:0.1:5;
Th=1./(1+exp(cD));
plot(cD,Th,'LineWidth',3);
grid on;
xlabel('|cD_i(k)|')
ylabel('\gamma_i');
set(gca,'FontSize',20);