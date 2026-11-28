function save_fig(f, name)
% Save a figure as 300 dpi PNG and vector PDF next to the scripts' parent folder.
out = fullfile(fileparts(mfilename('fullpath')), '..', name);
set(f, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
print(f, [out '.png'], '-dpng', '-r300');
pos = get(f, 'Position');
set(f, 'PaperUnits', 'points', 'PaperSize', pos(3:4), 'PaperPosition', [0 0 pos(3:4)]);
print(f, [out '.pdf'], '-dpdf');
end
