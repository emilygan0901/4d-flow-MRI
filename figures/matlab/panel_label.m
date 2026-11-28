function panel_label(ax, letter, titleText, dx)
% Bold panel letter and a plain title above the top-left corner of an axes.
% dx: horizontal offset of the letter in normalised axes units (default -0.16).
if nargin < 4, dx = -0.16; end
C = fig_colors();
pos = get(ax, 'Position'); gap = 0.03 / pos(3);   % constant gap in figure units
text(ax, dx, 1.08, letter, 'Units', 'normalized', 'FontName', C.font, 'FontSize', 13, ...
    'FontWeight', 'bold', 'Color', C.ink, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
text(ax, dx + gap, 1.08, titleText, 'Units', 'normalized', 'FontName', C.font, 'FontSize', 9.5, ...
    'Color', C.ink, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
end
