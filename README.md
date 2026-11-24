# 4D Flow MRI in Loeys-Dietz syndrome

Final-year project, Department of Bioengineering, Imperial College London.
Supervisor: Dr Muhammad Usman (Guy's and St Thomas' NHS Foundation Trust).

The project uses 4D Flow MRI of the thoracic aorta in 21 patients with
Loeys-Dietz syndrome and 12 healthy volunteers. It first replicates
established haemodynamic differences, then tests whether a compact
autoencoder learns flow features that differ between groups beyond aortic
diameter, and finally describes variation within the patient group.

> **No patient data are stored in this repository.** All imaging and
> derived data are processed only inside the Imperial College Secure
> Enclave. See `.gitignore`.

## Contents

| Path | What it is |
|---|---|
| `config/pipeline_config.yaml` | Locked processing settings (v1.0) with the phantom evidence for each choice |
| `validation/phantom/` | Digital phantom tests of the SlicerHeart 4D Flow processing steps |
| `validation/results/` | Saved outputs of those tests (JSON) |
| `validation/record_versions.py` | Writes Slicer, Python and package versions into the config |
| `figures/make_phantom_figures.py` | Builds the phantom figures (PNG, PDF, SVG) |
| `pipeline/planes.py` | Centreline frames (parallel transport) and 16 x 32 polar plane sampling |
| `pipeline/timing.py` | Normalisation to 20 cardiac phases and PWV transit-time estimation |

## Phantom validation

The tests use the digital phantoms of `slicerheart_4dflow` 0.5.2, which have
analytic ground truth. They are numerical phantoms, not scanned physical
phantoms.

| Test | Script | Phantom | Result |
|---|---|---|---|
| Background phase correction, five methods | `background_correction.py` | Chest, 5 seeds | Residual static-tissue error 15.16 mm/s uncorrected, 0.23 mm/s with WAR+MSAC (WRLS 0.21, OLS 0.89, MSAC 0.43, ARTO 0.46) |
| Phase unwrapping and stroke volume | `unwrapping_flow_wss.py` | Womersley tube, radius 12 mm, 2.5 mm voxels, SNR 1.8, 5 seeds | VENC 600 mm/s, 9.7% voxels wrapped: stroke-volume error -41.7% raw, -1.0% after Laplacian-then-temporal unwrapping |
| WSS against voxel size | `wss_voxel_size.py` | Womersley and steady tube, noise-free | Pulsatile peak WSS error -12% at 0.5 mm, -39% at 1 mm, -59% at 2 mm, -65% at 2.5 mm |

WSS is therefore reported descriptively only and is not used in the
hypothesis tests.

### Pipeline steps (tests 1-7)

| Test | Script | What is checked | Result | Proposed pass rule |
|---|---|---|---|---|
| 1. Velocity axes and sign | `test1_encoding_sign.py` | Stenosis phantom, 2.5 m/s VENC, SNR 12, corrupted by 6 known swaps and sign flips, 3 seeds | All 21 inputs (7 cases x 3 seeds) recovered exactly. Swaps and partial flips detected by `encoding_check` (divergence ratio 6.5-37). A whole-field flip is invisible to divergence (ratio 1.0) and was fixed by `expected_flow_sign` | Correct mapping every time: **pass** |
| 2. Flow conservation | `test2_flow_conservation.py` | Stenosis (30%) at 2.5 mm, SNR 12, 13 planes 5 mm apart, 5 seeds. Bifurcation 60/40 | Net flow within 3.6% of truth at every plane, SD between planes 1.2%, largest errors at the narrowing. Bifurcation parent vs daughters -0.6 ± 1.1%, split 0.600 (true 0.600) | < 5% between planes: **pass** |
| 3a. Polar sampling | `test3_plane_sampling.py` | Poiseuille tube oblique to the voxel grid, 16 x 32 polar grid | 1 mm noise free: axial profile RMSE 0.3% of peak, radial and circumferential 0.0%, flow -0.05%. 2.5 mm SNR 12: profile RMSE 2.7%, radial and circumferential 0.4%, flow -0.5% | In-plane < 2%, flow < 2%: **pass** |
| 3b. Local frame | `test3_plane_sampling.py` | Aorta-like centreline, 305 mm, out-of-plane arch, 0.3 mm jitter | Parallel transport: orthonormal to 1e-16, largest turn 4.1 deg between points, no flips. Frenet frame: 88 flips up to 180 deg | No flips: **pass** (Frenet rejected) |
| 4. 20-phase resampling | `test4_temporal_resampling.py` | 15-40 frames, RR 600-1000 ms, random trigger offset, 50 repeats | Mean flow within 0.1% for both methods. Peak flow: linear interpolation -1% (40 frames) to -7% mean, -12% worst (15 frames). Fourier interpolation within 0.3% | Mean flow < 2%: **pass**. Use Fourier for retrospectively gated data |
| 5. PWV timing | `test5_pwv_timing.py` | Known delay over 250 mm, 20 planes, 20-40 frames, 3% noise, 200 repeats | Median bias within ±4% at all settings. Spread grows with stiffness: IQR 11-12% at 4 m/s, 34-43% at 12 m/s. 20-plane regression roughly halves the spread of the two-plane method | Bias < 10%: **pass**. Precision limited for stiff aortas |
| 6. Mask sensitivity | `test6_mask_sensitivity.py` | Womersley tube at 2.5 mm, SNR 1.8, mask eroded or dilated by one voxel, 5 seeds | True mask: flow -1.0%. Eroded: flow -14.8%, radius -20%. Dilated: flow +2.7%, radius +15%. WSS moves by 15-40 points either way | Reported, no pass rule. Under-segmentation is the costly error |
| 7. Divergence-free filter | `test7_divergence_free_wss.py` | Same tube, filters fdm, relaxed, dfw | fdm cut velocity noise by 18% (44 to 37 mm/s) with no change in flow or WSS. dfw raised velocity error to 57 mm/s. WSS error stayed -62 to -65% with every filter, the same as noise free | Filtering does not fix WSS: **not adopted** |

Notes
- Test 4: the Fourier result uses a smooth synthetic waveform covering the
  whole RR interval. It applies to retrospectively gated data only; for
  prospective gating (part of the cycle missing) linear interpolation is used.
  Confirm the gating on the pilot case.
- Test 5: PWV is unbiased but imprecise at high PWV, because the transit time
  (21 ms at 12 m/s) is about one frame. Group comparisons remain valid;
  individual PWV values in stiff aortas are uncertain. This is reported as a
  limitation of the H1 PWV outcome.
- Test 6: missing one voxel of lumen loses about 15% of flow, while one extra
  voxel adds only about 3%, because the extra voxels carry only noise. A
  segmentation that errs slightly large is safer than one that errs small.
  Every mask is checked visually, and lumen radius normalises the polar grid,
  so radius errors also shift where each grid ring falls.
- Test 7: the WSS error is a voxel-size bias, not noise, so no filter removes
  it. WSS stays descriptive only.

### Reproduce

```bash
pip install -r requirements.txt
python validation/phantom/background_correction.py
python validation/phantom/unwrapping_flow_wss.py
python validation/phantom/wss_voxel_size.py
python validation/phantom/test1_encoding_sign.py
python validation/phantom/test2_flow_conservation.py
python validation/phantom/test3_plane_sampling.py
python validation/phantom/test4_temporal_resampling.py
python validation/phantom/test5_pwv_timing.py      # about 3 minutes
python validation/phantom/test6_mask_sensitivity.py
python validation/phantom/test7_divergence_free_wss.py
```

Results are written to `validation/results/`.

## Clinical dry run

`validation/clinical/dry_run.py` runs the locked settings on the 15 public,
anonymised clinical cases of the SlicerHeart project (CC BY 4.0, Siemens
Avanto fit, paediatric and congenital patients, 1.4-1.8 mm voxels, 12-30
frames). These are not study data. Aortic plane flows are compared with
Circle cvi42, measured independently by the data set authors
(`summarise_dry_run.py`, 29 planes).

| Check | Result |
|---|---|
| Cases processed | 15/15, 3-20 s each |
| Velocity encoding consistent | 15/15 |
| Gating coverage of the RR interval | 99-101% in all cases (retrospective), so Fourier phase normalisation applies |
| Net flow vs cvi42, locked settings | bias -0.7 mL (-2.6%), 95% limits -4.8 to +3.5 mL, mean absolute difference 12% |
| Net flow vs cvi42, SlicerHeart's own outputs | bias -1.0 mL, limits -4.5 to +2.4 mL, mean absolute difference 13% |
| Net flow vs cvi42, no background correction | bias +0.1 mL, limits -8.0 to +8.2 mL |
| Agreement by vessel radius | median absolute difference 9.9% below 5 mm radius, 5.9% above 8 mm |
| Polar grid flow vs plane flow | -2.6% median above 8 mm radius, -8.8% below 5 mm |

Background correction halves the limits of agreement; OLS, MSAC and WAR+MSAC
give the same result on these data. Agreement improves with vessel size,
and adult aortas (radius about 12-20 mm) are larger than any vessel here.
The polar grid assumes a circular lumen of equal area, so it misses some
flow in small or non-circular vessels: it is used as the input to the
autoencoder, and flow outcomes are measured with `plane_flow`.

## Settings policy

The settings in `config/pipeline_config.yaml` were fixed from the phantom
results before any patient data were processed. They are applied unchanged
to every subject. Any later change is recorded in the change log at the end
of the file, with the reason, and all subjects are re-processed.

## Next steps

- Batch pipeline and per-case QC report
