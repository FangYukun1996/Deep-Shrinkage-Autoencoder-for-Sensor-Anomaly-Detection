err=0:0.1:5;
Th=(1./(1+exp(err)))+1;
plot(err,Th,'LineWidth',3);
grid on;
xlabel('err')
ylabel('\beta');
axis([0 5 0 2])
set(gca,'FontSize',20);