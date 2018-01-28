function f1=recordTrajectory
% clear;
close all;
f1 = figure;
figure(f1);

axis([-2 2 -2 2])
hold on

% plot(-1,0, 'bo', 'MarkerSize', 40);
% plot(1,0,'ro','MarkerSize', 40);
% plot(0,1,'ro','MarkerSize', 40);

global trajectory
global mouseDown;
global recordFinished;

trajectory = [];
mouseDown = false;
recordFinished = false;

set(gcf, 'WindowButtonMotionFcn', @mouseMove);
set(gcf,'WindowButtonDownFcn', @record, 'WindowButtonUpFcn',   @stop);

end