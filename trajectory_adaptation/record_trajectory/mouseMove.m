function mouseMove(object, eventdata)
    global trajectory;
    global mouseDown;
    global recordFinished;
    
    C = get (gca, 'CurrentPoint');
    title(gca, ['(X,Y) = (', num2str(C(1,1)), ', ',num2str(C(1,2)), ')']);
    
    if mouseDown && ~recordFinished
        trajectory = [trajectory; C(1,1), C(1,2)];
        plot(trajectory(:,1), trajectory(:,2), 'b-');
    end
    assignin('base','trajectory',trajectory)
end