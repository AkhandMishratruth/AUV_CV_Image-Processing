y1 = 0:0.1:180;
x = 90;
figure(1);
axis([0,180,0,5]);
while(1)
    x = x + 0.5;
    if(x>=180)
        x = 0;
    end
    y2 = abs(1-abs(y1-x)/90);
    plot(y1,y2);
    pause(0.01);
end