function Result = Show( hObject, eventdata, handles , errorPlot,errorPlotE, axes_num , X_Data, Y_Data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if axes_num == 1
    axes(handles.axes1);

    plot(errorPlot(1,:), errorPlot(2,:),'-ro');
    hold on;
    plot(errorPlotE(1,:), errorPlotE(2,:),'-.b');
    xlabel(X_Data);
    ylabel(Y_Data);
    legend('Error Traning','Error evaluate');
    guidata(hObject, handles);
end
if axes_num == 2
    axes(handles.axes2);
      mesh(errorPlot)
    guidata(hObject, handles);
end

end
