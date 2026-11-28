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
% leave gaps between the three groups of factors
y = [17 16 15 14, 12 11, 9 8 7 6 5];
f = figure('Color', 'w', 'Units', 'points', 'Position', [50 50 520 260]);

ax = axes('Position', [0.19 0.17 0.4 0.68]); hold(ax, 'on');
for i = 1:n
    patch(ax, [0 R(i, 1) R(i, 1) 0], y(i) + 0.04 + [0 0 0.36 0.36], C.teal, 'EdgeColor', 'none');
    patch(ax, [0 R(i, 2) R(i, 2) 0], y(i) - 0.04 - [0 0 0.36 0.36], C.grey, 'EdgeColor', 'none');
    if i <= 4
        plot(ax, R(i, 3) * [1 1], y(i) + [-0.02 0.46], '-', 'Color', C.ink, 'LineWidth', 2);
    end
end
gy = [17.9 12.9 9.9]; gl = {'Haemodynamics', 'Geometry', 'Acquisition'};
for g = 1:3
    text(ax, 1, gy(g), gl{g}, 'HorizontalAlignment', 'right', 'FontName', C.font, 'FontSize', 7.5, ...
        'FontAngle', 'italic', 'Color', C.axis);
end
plot(ax, [0 1], [13.5 13.5], '-', 'Color', C.edge, 'LineWidth', 0.6);
plot(ax, [0 1], [10.5 10.5], '-', 'Color', C.edge, 'LineWidth', 0.6);
text(ax, 0.03, 9.0, 'All ≈ 0, so acquisition is not encoded', 'FontName', C.font, 'FontSize', 7.5, 'Color', C.ink);
set(ax, 'YTick', fliplr(y), 'YTickLabel', fliplr(labs), 'YLim', [4.3 18.4], 'XLim', [0 1], 'XTick', 0:0.2:1);
xlabel(ax, 'Variance explained by the 8 features (cross-validated R^2)', 'Color', C.ink);
hA = patch(ax, NaN, NaN, C.teal, 'EdgeColor', 'none'); hP = patch(ax, NaN, NaN, C.grey, 'EdgeColor', 'none');
hG = plot(ax, NaN, NaN, '-', 'Color', C.ink, 'LineWidth', 2);
lg = legend(ax, [hA hP hG], {'Autoencoder', 'PCA', 'Autoencoder, local geometry removed'}, 'Box', 'off', ...
    'Color', 'none', 'FontSize', 7.5);
set(lg, 'Position', [0.3 0.19 0.25 0.14]);
style_axes(ax); set(ax, 'TickLength', [0 0]);
panel_label(ax, 'A', 'What the features capture', -0.42);

P = read_csv('synthetic_power.csv');
ax = axes('Position', [0.7 0.17 0.27 0.68]); hold(ax, 'on');
patch(ax, [-0.05 1.65 1.65 -0.05], [0 0 0.05 0.05], C.band, 'EdgeColor', 'none');
plot(ax, [-0.05 1.65], [0.05 0.05], '--', 'Color', C.pink, 'LineWidth', 0.9);
plot(ax, [-0.05 1.65], [0.8 0.8], ':', 'Color', C.axis, 'LineWidth', 0.9);
text(ax, 1.62, 0.075, '5% false-positive level', 'HorizontalAlignment', 'right', 'FontName', C.font, 'FontSize', 7, 'Color', C.pink);
text(ax, 1.62, 0.83, '80% target power', 'HorizontalAlignment', 'right', 'FontName', C.font, 'FontSize', 7, 'Color', C.axis);
p1 = plot(ax, P(:, 1), P(:, 2), '-o', 'Color', C.teal, 'LineWidth', 1.6, 'MarkerSize', 5, 'MarkerFaceColor', C.teal);
p2 = plot(ax, P(:, 1), P(:, 3), '-s', 'Color', C.grey, 'LineWidth', 1.6, 'MarkerSize', 5, 'MarkerFaceColor', C.grey);
xlim(ax, [-0.05 1.65]); ylim(ax, [0 1]); set(ax, 'YTick', 0:0.2:1, 'YTickLabel', {'0', '20', '40', '60', '80', '100'});
xlabel(ax, {'Planted helicity difference,', 'LDS vs volunteers (Cohen''s d)'}, 'Color', C.ink);
ylabel(ax, 'Cohorts with a significant feature (%)', 'Color', C.ink);
lg = legend(ax, [p1 p2], {'Autoencoder', 'PCA'}, 'Box', 'off', 'Color', 'none', 'FontSize', 7.5);
set(lg, 'Position', [0.71 0.6 0.12 0.1]);
style_axes(ax);
panel_label(ax, 'B', 'Planted 21 vs 12 cohorts', -0.3);
save_fig(f, 'Figure_synthetic_recovery_matlab');
