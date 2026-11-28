# MATLAB report figures

Publication versions of Figures 7–9 of the planning report, in the same colours as the other figures.

1. `python figures/matlab/export_csv.py` (from the repository root) writes the result tables in `tables/`.
   They hold only phantom, synthetic and public CC BY 4.0 results, no study data.
2. In MATLAB, `cd figures/matlab` and run
   - `Figure_phantom_validation` (Figure 7)
   - `Figure_clinical_agreement` (Figure 8)
   - `Figure_synthetic_recovery` (Figure 9)

Each script saves a 300 dpi PNG and a vector PDF to `figures/` with the suffix `_matlab`.
No toolboxes are needed, and the scripts also run in GNU Octave 8+.
Shared style lives in `fig_colors.m`, `style_axes.m` and `panel_label.m`.
