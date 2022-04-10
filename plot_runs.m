function plot_runs(csvs, names, titletext, smooth, cutoff)
    figure('NumberTitle', 'off', 'Name', titletext);
    for i = 1:length(names)
        Array=csvread(csvs{i});
        col1 = Array(1:16, 2);
        col2 = Array(1:16, 3);
        if i == 3
            col1 = Array(1:5, 2);
            col2 = Array(1:5, 3);
        end
        if i == 2
            col1 = Array(1:5, 2);
            col2 = Array(1:5, 3);
        end
        col2 = smoothdata(col2, 'movmedian', smooth);
        plot(col1, col2, 'LineWidth', 2.5)
        hold on;
    end
    
    legend(names, 'Location', 'southeast', 'FontSize', 18)
    title(titletext)
    xlabel('steps')

end