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
| `pipeline/` | Batch processing pipeline (in development) |

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

### Reproduce

```bash
pip install -r requirements.txt
python validation/phantom/background_correction.py
python validation/phantom/unwrapping_flow_wss.py
python validation/phantom/wss_voxel_size.py
```

Results are written to `validation/results/`.

## Settings policy

The settings in `config/pipeline_config.yaml` were fixed from the phantom
results before any patient data were processed. They are applied unchanged
to every subject. Any later change is recorded in the change log at the end
of the file, with the reason, and all subjects are re-processed.

## Next steps

- Phantom tests of velocity encoding and sign, flow conservation along the
  vessel, plane sampling and local frame, temporal resampling, PWV timing,
  mask sensitivity and divergence-free filtering
- Dry run on the public `slicerheart_4dflow` clinical test cases
- Batch pipeline and per-case QC report
