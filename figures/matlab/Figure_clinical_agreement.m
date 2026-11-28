% Figure 8. Bland-Altman agreement of aortic net flow with Circle cvi42.
% (A) locked pipeline (WAR+MSAC background correction), (B) without background
% correction. Points shaded by lumen radius. Data: tables/clinical_agreement.csv
% Run in MATLAB (or Octave) from this folder:  Figure_clinical_agreement
C = fig_colors();
M = read_csv('clinical_agreement.csv');
ref = M(:, 1); rad = M(:, 4);
groups = {rad < 5, rad >= 5 & rad <= 8, rad > 8};
gcol = {C.light, C.teal, C.dark}; glab = {'< 5 mm', '5–8 mm', '> 8 mm'};
f = figure('Color', 'w', 'Units', 'points', 'Position', [50 50 520 230]);
titles = {'Locked pipeline (WAR+MSAC)', 'Without background correction'};
for k = 1:2
    m = M(:, 1 + k);
    mn = (m + ref) / 2; d = m - ref;
    b = mean(d); s = std(d); lo = b - 1.96 * s; hi = b + 1.96 * s;
    ax = axes('Position', [0.09 + (k - 1) * 0.47, 0.25, 0.38, 0.6]); hold(ax, 'on');
    patch(ax, [-85 85 85 -85], [lo lo hi hi], C.band, 'EdgeColor', 'none');
    plot(ax, [-85 85], [0 0], '-', 'Color', C.edge, 'LineWidth', 0.7);
    plot(ax, [-85 85], [b b], '-', 'Color', C.ink, 'LineWidth', 1.2);
    plot(ax, [-85 85], [lo lo], '--', 'Color', C.ink, 'LineWidth', 0.9);
    plot(ax, [-85 85], [hi hi], '--', 'Color', C.ink, 'LineWidth', 0.9);
    h = zeros(1, 3);
    for g = 1:3
        h(g) = scatter(ax, mn(groups{g}), d(groups{g}), 32, gcol{g}, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
    end
    labs = {sprintf('+1.96 SD  %+.1f', hi), sprintf('Bias  %+.1f', b), sprintf('−1.96 SD  %+.1f', lo)};
    ys = [hi b lo];
    for j = 1:3
        text(ax, 83, ys(j) + 0.45, labs{j}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
            'FontName', C.font, 'FontSize', 7.5, 'Color', C.ink);
    end
    xlim(ax, [-85 85]); ylim(ax, [-12 17.5]);
    xlabel(ax, 'Mean of pipeline and cvi42 (mL per beat)', 'Color', C.ink);
    if k == 1, ylabel(ax, 'Pipeline − cvi42 (mL per beat)', 'Color', C.ink); end
    style_axes(ax);
    panel_label(ax, char('A' + k - 1), titles{k}, -0.2);
end
lg = legend(ax, h, glab, 'Orientation', 'horizontal', 'Box', 'off', 'Color', 'none', 'FontName', C.font, 'FontSize', 8);
set(lg, 'Position', [0.34 0.015 0.36 0.06]);
fig_text(f, 0.33, 0.045, 'Lumen radius', 'Color', C.axis, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
save_fig(f, 'Figure_clinical_agreement_matlab');
