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
    plot(ax, [v(i) - s(i), v(i) + s(i)], [y(i) y(i)], '-', 'Color', C.axis, 'LineWidth', 0.8);
    text(ax, (v(i) + s(i)) * 1.12, y(i), sprintf('%.2f', v(i)), 'FontName', C.font, 'FontSize', 8, ...
        'Color', C.ink, 'VerticalAlignment', 'middle');
end
% WRLS is marginally lower, so state why WAR+MSAC is the locked choice
text(ax, 0.62, y(n), 'selected, also rejects wrap-around', 'FontName', C.font, 'FontSize', 7.5, ...
    'FontAngle', 'italic', 'Color', C.dark, 'VerticalAlignment', 'middle');
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
text(ax, 2.5, 13, {'Clinical voxels', '(2–3 mm)'}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
    'FontName', C.font, 'FontSize', 7, 'Color', C.axis);
plot(ax, [0.3 3.0], [0 0], '-', 'Color', C.axis, 'LineWidth', 0.9);
text(ax, 2.96, -98, 'Negative = underestimation', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'FontName', C.font, ...
    'FontSize', 7, 'Color', C.axis);
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

% D  Pulse wave velocity ----------------------------------------------------
P = read_csv('phantom_pwv.csv');
tv = unique(P(:, 1))'; ax = axes('Position', [X(2) Y(2) W H]); hold(ax, 'on');
cols = {C.grey, C.teal}; off = [-0.22 0.22]; bw = 0.36; hb = zeros(1, 2);
for k = 1:numel(tv)
    for m = 1:2
        e = P(P(:, 1) == tv(k) & P(:, 2) == m, 3);
        q = pctl(e, [5 25 50 75 95]); x0 = k + off(m);
        plot(ax, [x0 x0], q([1 2]), '-', 'Color', cols{m}, 'LineWidth', 1.3);
        plot(ax, [x0 x0], q([4 5]), '-', 'Color', cols{m}, 'LineWidth', 1.3);
        plot(ax, x0 + [-0.08 0.08], q([1 1]), '-', 'Color', cols{m}, 'LineWidth', 1.3);
        plot(ax, x0 + [-0.08 0.08], q([5 5]), '-', 'Color', cols{m}, 'LineWidth', 1.3);
        hb(m) = patch(ax, x0 + bw / 2 * [-1 1 1 -1], q([2 2 4 4]), cols{m}, 'EdgeColor', cols{m}, 'LineWidth', 1.2, 'FaceAlpha', 0.85);
        plot(ax, x0 + bw / 2 * [-1 1], q([3 3]), '-', 'Color', C.ink, 'LineWidth', 2);
    end
end
% true value drawn last, in the reference colour, so it stays visible over the medians
for k = 1:numel(tv)
    ht = plot(ax, k + [-0.5 0.5], [tv(k) tv(k)], '--', 'Color', C.pink, 'LineWidth', 1.6);
end
set(ax, 'XTick', 1:numel(tv), 'XTickLabel', arrayfun(@num2str, tv, 'UniformOutput', false));
xlim(ax, [0.4 numel(tv) + 0.6]); ylim(ax, [0 30]);
xlabel(ax, 'True PWV (m/s)', 'Color', C.ink); ylabel(ax, 'Estimated PWV (m/s)', 'Color', C.ink);
legend(ax, [hb ht], {'Two planes', '20-plane regression', 'True PWV'}, 'Box', 'off', 'Color', 'none', 'FontSize', 7.5, 'Location', 'northwest');
style_axes(ax);
panel_label(ax, 'D', 'Pulse wave velocity', -0.22);

save_fig(f, 'Figure_phantom_validation_matlab');
