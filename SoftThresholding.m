x=-3:3;
y=[-2,-1,0,0,0,1,2];
plot(x,y,'LineWidth',3);
grid on;
xlabel('x')
ylabel('y');
set(gca,'XTickLabel',{'-3\pi' '-2\pi' '-\pi' '0' '\pi' '2\pi' '3\pi'});
set(gca,'YTickLabel',{'-2\pi' '-\pi' '0' '\pi' '2\pi'});
set(gca,'FontSize',20);