# Release Notes

## Suggested release title

`Type II surrogate checkpoints v1`

## Suggested tag

`typeii-checkpoints-v1`

## Summary

This release provides the external checkpoint files required by the retained
Type II surrogate inference pathways in `redback_surrogates/supernovamodels.py`.

Included in this release:

- interaction-model checkpoint
- photospheric-model autoencoder checkpoint
- photospheric-model emulator checkpoint
- checksum manifest for verification

This release does not include direct-regression checkpoints.

## Assets to upload

Upload these files to the GitHub Release without renaming them:

- `emulator_6param_timeweighted_best.pt`
- `ae_cnn_v3_best.pt`
- `emulator_cnn_v3_6param_best.pt`
- `sha256sums.txt`

## Checksums

- `emulator_6param_timeweighted_best.pt`
  - SHA256: `9e799e8082b21b8bc00ab3c2f7efd53198414ae0e7718523583fbf907d46b4e8`
  - Size: `249418250` bytes
- `ae_cnn_v3_best.pt`
  - SHA256: `1a6ad87e31cf1e13acd19d2d49221710c7cbee2da034e135129ad8ad23c0031d`
  - Size: `46381034` bytes
- `emulator_cnn_v3_6param_best.pt`
  - SHA256: `b1326e471432288cebf79638924045ddb42638dcc31390decaae0e7763c4514e`
  - Size: `8995756` bytes

## Installation note

The current code paths support external checkpoint directories via:

- `STELLA_INTERACTION_MODEL_DIR`
- `STELLA_PHOTOSPHERIC_MODEL_DIR`

Users should place the downloaded files into:

```text
<checkpoint-root>/interaction_model/emulator_6param_timeweighted_best.pt
<checkpoint-root>/photospheric_model/ae_cnn_v3_best.pt
<checkpoint-root>/photospheric_model/emulator_cnn_v3_6param_best.pt
```

## Suggested release body

```md
This release provides the external checkpoint files required by the retained
Type II surrogate inference pathways in `redback_surrogates`.

Included assets:

- `emulator_6param_timeweighted_best.pt`
- `ae_cnn_v3_best.pt`
- `emulator_cnn_v3_6param_best.pt`
- `sha256sums.txt`

These checkpoints are intended for the interaction and photospheric surrogate
pathways only. Direct-regression checkpoints are not included in this release.

The corresponding code can locate these files through:

- `STELLA_INTERACTION_MODEL_DIR`
- `STELLA_PHOTOSPHERIC_MODEL_DIR`

Please verify downloaded files with `sha256sums.txt` before use.
```

## GitHub Release checklist

- create tag `typeii-checkpoints-v1`
- upload the three `.pt` files
- upload `sha256sums.txt`
- paste the release body above
- test at least one download URL after publishing
- if Zenodo integration is enabled, wait for the linked record to appear
