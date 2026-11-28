% Figure 7. Validation of the processing steps on digital phantoms with analytic
% ground truth. (A) background phase correction, (B) phase unwrapping,
% (C) WSS error against voxel size, (D) PWV from two planes and 20-plane regression.
% Data: tables/phantom_*.csv (exported by export_csv.py from validation/results).
% Run in MATLAB (or Octave) from this folder:  Figure_phantom_validation
C = fig_colors();
f = figure('Color', 'w', 'Units', 'points', 'Position', [50 50 520 440]);
W = 0.36; H = 0.31; X = [0.15 0.62]; Y = [0.58 0.11];

% A  Background phase correction -------------------------------------------
B = read_csv('phantom_background.csv');
names = {'None', 'OLS', 'ARTO', 'MSAC', 'WRLS', 'WAR+MSAC'};
v = B(:, 2); s = B(:, 3); n = numel(v); y = n:-1:1;
ax = axes('Position', [X(1) Y(1) W H]); hold(ax, 'on');
for i = 1:n
    c = C.grey; if i == 1, c = C.pink; elseif i == n, c = C.teal; end
    patch(ax, [0.1 v(i) v(i) 0.1], y(i) + [-0.32 -0.32 0.32 0.32], c, 'EdgeColor', 'none');
    text(ax, v(i) * 1.12, y(i), sprintf('%.2f', v(i)), 'FontName', C.font, 'FontSize', 8, ...
        'Color', C.ink, 'VerticalAlignment', 'middle');
end
set(ax, 'XScale', 'log', 'XLim', [0.1 40], 'YLim', [0.4 n + 0.6], 'YTick', 1:n, 'YTickLabel', fliplr(names), ...
    'XTick', [0.1 1 10], 'XTickLabel', {'0.1', '1', '10'});
xlabel(ax, 'Residual velocity in static tissue (mm/s, log scale)', 'Color', C.ink);
style_axes(ax); set(ax, 'TickLength', [0 0]);
panel_label(ax, 'A', 'Background phase correction', -0.3);

% B  Phase unwrapping -------------------------------------------------------
U = read_csv('phantom_unwrap.csv');
ax = axes('Position', [X(2) Y(1) W H]); hold(ax, 'on');
h1 = plot(ax, U(:, 1), U(:, 2), '-', 'Color', C.ink, 'LineWidth', 2.2);
h2 = plot(ax, U(:, 1), U(:, 3), '-o', 'Color', C.pink, 'LineWidth', 1.2, 'MarkerSize', 3.5, 'MarkerFaceColor', C.pink);
h3 = plot(ax, U(:, 1), U(:, 4), '--s', 'Color', C.teal, 'LineWidth', 1.2, 'MarkerSize', 3.5, 'MarkerFaceColor', C.teal);
xlim(ax, [-20 780]); ylim(ax, [0 360]); set(ax, 'YTick', 0:50:300);
xlabel(ax, 'Time in cardiac cycle (ms)', 'Color', C.ink); ylabel(ax, 'Flow (mL/s)', 'Color', C.ink);
lg = legend(ax, [h1 h2 h3], {'True flow', 'Aliased flow', 'Unwrapped flow'}, 'Box', 'off', 'Color', 'none', 'FontSize', 7.5, 'Location', 'northeast');
style_axes(ax);
panel_label(ax, 'B', 'Phase unwrapping', -0.22);

% C  Wall shear stress against voxel size ----------------------------------
Wd = read_csv('phantom_wss.csv');
ax = axes('Position', [X(1) Y(2) W H]); hold(ax, 'on');
% clinical 4D Flow voxels are about 2-3 mm: very light band, labelled directly
patch(ax, [2.0 3.0 3.0 2.0], [-100 -100 15 15], [0.965 0.972 0.978], 'EdgeColor', 'none');
plot(ax, [0.3 3.0], [0 0], ':', 'Color', C.edge, 'LineWidth', 0.7);
text(ax, 2.5, -97, {'Clinical', 'voxels'}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'FontName', C.font, 'FontSize', 7, 'Color', C.axis);
g1 = plot(ax, Wd(:, 1), Wd(:, 2), '-o', 'Color', C.teal, 'LineWidth', 1.4, 'MarkerSize', 4, 'MarkerFaceColor', C.teal);
g2 = plot(ax, Wd(:, 1), Wd(:, 3), '-o', 'Color', C.pink, 'LineWidth', 1.4, 'MarkerSize', 4, 'MarkerFaceColor', C.pink);
g3 = plot(ax, Wd(:, 1), Wd(:, 4), '--v', 'Color', C.pink, 'LineWidth', 1.4, 'MarkerSize', 4, 'MarkerFaceColor', C.pink);
text(ax, 2.58, Wd(end, 4), sprintf('%.0f%%', Wd(end, 4)), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
    'FontName', C.font, 'FontSize', 7.5, 'Color', C.pink, 'FontWeight', 'bold');
xlim(ax, [0.3 3.0]); ylim(ax, [-100 15]); set(ax, 'XTick', 0.5:0.5:2.5, 'XTickLabel', {'0.5','1.0','1.5','2.0','2.5'}, 'YTick', -100:20:0);
xlabel(ax, 'Voxel size (mm)', 'Color', C.ink); ylabel(ax, 'Error in WSS relative to truth (%)', 'Color', C.ink);
legend(ax, [g1 g2 g3], {'Steady flow', 'Pulsatile, time-averaged', 'Pulsatile, peak systole'}, ...
    'Box', 'off', 'Color', 'none', 'FontSize', 7.5, 'Location', 'southwest');
style_axes(ax);
panel_label(ax, 'C', 'Wall shear stress', -0.3);

% D  Pulse wave velocity: estimated against true, line of identity -----------
% Conventional validation layout: median estimate with interquartile-range
% error bars for each method, both methods offset slightly about the true value.
P = read_csv('phantom_pwv.csv');
tv = unique(P(:, 1))'; ax = axes('Position', [X(2) Y(2) W H]); hold(ax, 'on');
hid = plot(ax, [0 15], [0 15], '--', 'Color', C.edge, 'LineWidth', 1);
cols = {C.grey, C.teal}; mk = {'s', 'o'}; off = [-0.22 0.22]; hm = zeros(1, 2);
for m = 1:2
    med = zeros(size(tv)); lo = med; hi = med;
    for k = 1:numel(tv)
        q = pctl(P(P(:, 1) == tv(k) & P(:, 2) == m, 3), [25 50 75]);
        lo(k) = q(1); med(k) = q(2); hi(k) = q(3);
    end
    x = tv + off(m);
    for k = 1:numel(tv)
        plot(ax, [x(k) x(k)], [lo(k) hi(k)], '-', 'Color', cols{m}, 'LineWidth', 1.4);
        plot(ax, x(k) + [-0.12 0.12], [lo(k) lo(k)], '-', 'Color', cols{m}, 'LineWidth', 1.4);
        plot(ax, x(k) + [-0.12 0.12], [hi(k) hi(k)], '-', 'Color', cols{m}, 'LineWidth', 1.4);
    end
    hm(m) = plot(ax, x, med, mk{m}, 'MarkerSize', 6, 'MarkerFaceColor', cols{m}, 'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
end
xlim(ax, [2 14]); ylim(ax, [2 18]); set(ax, 'XTick', tv, 'YTick', 4:4:16);
xlabel(ax, 'True PWV (m/s)', 'Color', C.ink); ylabel(ax, 'Estimated PWV (m/s)', 'Color', C.ink);
lg = legend(ax, [hm hid], {'Two planes', '20-plane regression', 'Line of identity'}, 'Box', 'off', 'Color', 'none', 'FontSize', 7.5);
set(lg, 'Position', [X(2) + 0.19, Y(2) + 0.008, 0.13, 0.1]);
style_axes(ax);
panel_label(ax, 'D', 'Pulse wave velocity', -0.22);

save_fig(f, 'Figure_phantom_validation_matlab');
