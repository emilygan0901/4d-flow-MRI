% Figure 9. Synthetic test of feature learning on 300 3D aortas.
% (A) variance in each known factor explained by the 8 learned features
% (cross-validated R2, non-linear probe), (B) detection rate in planted
% 21 vs 12 cohorts against the planted helicity difference.
% Data: tables/synthetic_recovery.csv, tables/synthetic_power.csv
% Run in MATLAB (or Octave) from this folder:  Figure_synthetic_recovery
C = fig_colors();
R = read_csv('synthetic_recovery.csv'); R(R < 0) = 0;
labs = {'Mean velocity', 'Helicity', 'Jet eccentricity', 'Flow reversal', 'Dean (curvature)', ...
    'Lumen radius', 'Voxel size', 'Frames', 'VENC', 'SNR', 'Background offset'};
n = numel(labs);
% Horizontal bars on evenly spaced rows (one row per factor, no extra gaps);
% groups are marked by thin separators and rotated labels at the left.
y = n:-1:1;
f = figure('Color', 'w', 'Units', 'points', 'Position', [50 50 520 270]);

% A  What the features capture --------------------------------------------
ax = axes('Position', [0.215 0.23 0.375 0.64]); hold(ax, 'on');
bh = 0.36;   % height of each bar; the pair fills most of the row
for i = 1:n
    patch(ax, [0 R(i, 1) R(i, 1) 0], y(i) + [0 0 bh bh], C.teal, 'EdgeColor', 'none');
    patch(ax, [0 R(i, 2) R(i, 2) 0], y(i) - [0 0 bh bh], C.grey, 'EdgeColor', 'none');
    if R(i, 1) < 0.02   % value near zero: small tick so it reads as a result
        plot(ax, [0 0.012], y(i) + bh / 2 * [1 1], '-', 'Color', C.teal, 'LineWidth', 2.5);
    end
    if R(i, 2) < 0.02
        plot(ax, [0 0.012], y(i) - bh / 2 * [1 1], '-', 'Color', C.grey, 'LineWidth', 2.5);
    end
end
hG = plot(ax, R(1:4, 3), y(1:4)' + bh / 2, 'd', 'MarkerSize', 6, 'MarkerFaceColor', C.ink, 'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
sep = [mean(y(4:5)) mean(y(6:7))];
for k = 1:2
    plot(ax, [-0.5 1], [sep(k) sep(k)], '-', 'Color', C.edge, 'LineWidth', 0.6, 'Clipping', 'off');
end
grp = {1:4, 5:6, 7:11}; gl = {'Haemodynamics', 'Geometry', 'Acquisition'};
for g = 1:3
    yy = mean(y(grp{g}));
    text(ax, -0.5, yy, gl{g}, 'Rotation', 90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontName', C.font, 'FontSize', 6.8, 'FontAngle', 'italic', 'Color', C.axis);
end
set(ax, 'YTick', fliplr(y), 'YTickLabel', fliplr(labs), 'YLim', [0.45 n + 0.55], 'XLim', [0 1], 'XTick', 0:0.2:1);
xlabel(ax, 'Cross-validated R^2', 'Color', C.ink);
style_axes(ax); set(ax, 'TickLength', [0 0]);
panel_label(ax, 'A', 'What the features capture', -0.52);
hA = patch(ax, NaN, NaN, C.teal, 'EdgeColor', 'none'); hP = patch(ax, NaN, NaN, C.grey, 'EdgeColor', 'none');
lg = legend(ax, [hA hP hG], {'Autoencoder', 'PCA', 'Autoencoder, local geometry removed'}, 'Box', 'off', ...
    'Color', 'none', 'FontSize', 7.5, 'Orientation', 'horizontal');
set(lg, 'Position', [0.06 0.01 0.6 0.06]);

% B  Planted 21 vs 12 cohorts ---------------------------------------------
P = read_csv('synthetic_power.csv');
ax = axes('Position', [0.69 0.23 0.27 0.64]); hold(ax, 'on');
plot(ax, [-0.05 1.65], [0.05 0.05], '--', 'Color', 0.55 * C.pink + 0.45, 'LineWidth', 0.6);
plot(ax, [-0.05 1.65], [0.8 0.8], ':', 'Color', C.edge, 'LineWidth', 0.8);
text(ax, 1.62, 0.085, '5% false-positive level', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
    'FontName', C.font, 'FontSize', 7, 'Color', C.pink);
text(ax, 1.62, 0.83, '80% target power', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
    'FontName', C.font, 'FontSize', 7, 'Color', C.axis);
dx = 0.025;   % small horizontal offset so overlapping points stay visible
p1 = plot(ax, P(:, 1) - dx, P(:, 2), '-o', 'Color', C.teal, 'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', C.teal, 'MarkerEdgeColor', 'w');
p2 = plot(ax, P(:, 1) + dx, P(:, 3), '-s', 'Color', C.grey, 'LineWidth', 1.6, 'MarkerSize', 7, 'MarkerFaceColor', C.grey, 'MarkerEdgeColor', 'w');
xlim(ax, [-0.05 1.65]); ylim(ax, [0 1]); set(ax, 'YTick', 0:0.2:1, 'YTickLabel', {'0', '20', '40', '60', '80', '100'});
xlabel(ax, 'Planted helicity difference (Cohen''s d)', 'Color', C.ink);
ylabel(ax, 'Cohorts with a significant feature (%)', 'Color', C.ink);
lg2 = legend(ax, [p1 p2], {'Autoencoder', 'PCA'}, 'Box', 'off', 'Color', 'none', 'FontSize', 7.5);
set(lg2, 'Position', [0.71 0.62 0.12 0.1]);
style_axes(ax);
panel_label(ax, 'B', 'Planted 21 vs 12 cohorts', -0.3);
save_fig(f, 'Figure_synthetic_recovery_matlab');
