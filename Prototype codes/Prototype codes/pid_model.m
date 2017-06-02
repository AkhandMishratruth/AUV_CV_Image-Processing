error = 5.0;
kp = 1.0;
ki = 0.5;
kd = 0.5;
dt = 0.005;
figure(3), plot(0,error,'g.');
axis([0,20,-10,10]);
hold on;
grid;
time = 0;
last_error = error;
integral = 0;
while (1)
    %speed_error = kp * error + ki * integral*dt - kd * (error-last_error)/dt;
    speed_error = kp * error + ki * sign(error) - kd * (error-last_error)/dt;
    time = time + dt;
    pause(dt);
    integral = integral + error;
    last_error = error;
    error = error - speed_error*dt;
    plot(time,error,'g.');
end