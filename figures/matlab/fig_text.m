function fig_text(f, x, y, str, varargin)
% Text placed in figure-normalised coordinates (invisible full-figure axes).
C = fig_colors();
a = axes('Parent', f, 'Position', [0 0 1 1], 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], 'HitTest', 'off');
text(a, x, y, str, 'FontName', C.font, 'FontSize', 8, 'Color', C.ink, varargin{:});
end
