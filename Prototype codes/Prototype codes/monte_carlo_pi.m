trial = double(0);
hit = double(0);
%figure(1), axis([0,50000,0,5]),hold on;
%grid;
%figure(2), axis([0,1,0,1]),hold on;
r = 1;
xc = 0;
yc = 0;

theta = linspace(0,pi/2);
x = r*cos(theta) + xc;
y = r*sin(theta) + yc;
%plot(x,y,'k')
while(1)
    x = rand();
    y = rand();
    trial = trial + 1;
    if (x*x + y*y <= 1)
        hit = hit + 1;
    end
    %figure(2), plot(x,y,'b.');
    %figure(1), plot(trial,4*hit/trial,'r.');
    %pause(0.001);
    if trial>realmax-1
        break;
    end
end
disp(4*hit/trial)