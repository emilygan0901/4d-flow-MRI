function C = fig_colors()
% Colour palette shared by all report figures (same as the study schematic).
C.teal  = [62 155 166] / 255;   % primary: this pipeline / autoencoder
C.pink  = [232 67 107] / 255;   % contrast: problem condition / reference
C.green = [125 179 67] / 255;
C.grey  = [154 161 169] / 255;  % secondary: alternatives / PCA
C.light = [169 211 216] / 255;  % light teal
C.dark  = [29 90 98] / 255;     % dark teal
C.ink   = [31 35 40] / 255;     % text
C.axis  = [90 98 112] / 255;    % axes and ticks
C.edge  = [191 196 201] / 255;  % reference lines
C.band  = [238 241 243] / 255;  % shaded bands
C.font  = 'Arial';
end
