function style_axes(ax)
% Clean, journal-style axes: no box, outward ticks, grey axis lines, Arial.
C = fig_colors();
set(ax, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 0.8, 'FontName', C.font, 'FontSize', 8.5, ...
    'XColor', C.axis, 'YColor', C.axis, 'TickLength', [0.015 0.015], 'Layer', 'top');
end
