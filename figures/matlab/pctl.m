function q = pctl(x, p)
% Percentiles by linear interpolation (no toolbox needed). p in 0-100.
x = sort(x(~isnan(x))); n = numel(x);
pos = 1 + (n - 1) * p / 100; lo = floor(pos); hi = ceil(pos);
q = x(lo) + (pos - lo) .* (x(hi) - x(lo));
end
