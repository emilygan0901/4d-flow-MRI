function M = read_csv(name)
% Numeric CSV with one header line, from the data folder.
M = dlmread(fullfile(fileparts(mfilename('fullpath')), 'tables', name), ',', 1, 0);
end
