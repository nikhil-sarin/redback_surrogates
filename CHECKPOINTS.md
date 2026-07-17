# Type II Checkpoints

This repository does not bundle the Type II surrogate checkpoint files in the source tree.
The retained Type II surrogate pathways in `redback_surrogates/supernovamodels.py` expect
external `.pt` files for the interaction and photospheric models.

This document is a release-preparation note for distributing those checkpoints via
GitHub Releases on `zheng-yang-zhang/redback_surrogates`.

If you publish the checkpoints from a different repository or under a different tag,
update the URLs and tag names below before sharing the release publicly.

## Suggested release

- Repository: `https://github.com/zheng-yang-zhang/redback_surrogates`
- Suggested tag: `typeii-checkpoints-v1`
- Suggested download root:
  `https://github.com/zheng-yang-zhang/redback_surrogates/releases/download/typeii-checkpoints-v1`

## Required files

### Interaction model

- Filename: `emulator_6param_timeweighted_best.pt`
- Expected SHA256: `9e799e8082b21b8bc00ab3c2f7efd53198414ae0e7718523583fbf907d46b4e8`
- Size: `249418250` bytes
- Release URL:
  `https://github.com/zheng-yang-zhang/redback_surrogates/releases/download/typeii-checkpoints-v1/emulator_6param_timeweighted_best.pt`

### Photospheric model

- Filename: `ae_cnn_v3_best.pt`
- Expected SHA256: `1a6ad87e31cf1e13acd19d2d49221710c7cbee2da034e135129ad8ad23c0031d`
- Size: `46381034` bytes
- Release URL:
  `https://github.com/zheng-yang-zhang/redback_surrogates/releases/download/typeii-checkpoints-v1/ae_cnn_v3_best.pt`

- Filename: `emulator_cnn_v3_6param_best.pt`
- Expected SHA256: `b1326e471432288cebf79638924045ddb42638dcc31390decaae0e7763c4514e`
- Size: `8995756` bytes
- Release URL:
  `https://github.com/zheng-yang-zhang/redback_surrogates/releases/download/typeii-checkpoints-v1/emulator_cnn_v3_6param_best.pt`

## Recommended local layout

The code expects the original filenames, so the simplest installation pattern is:

```text
<checkpoint-root>/
  interaction_model/
    emulator_6param_timeweighted_best.pt
  photospheric_model/
    ae_cnn_v3_best.pt
    emulator_cnn_v3_6param_best.pt
```

For example:

```text
~/.cache/redback_surrogates/TypeII_Moriya/
  interaction_model/
  photospheric_model/
```

## Environment variables

The current implementation supports these overrides:

- `STELLA_INTERACTION_MODEL_DIR`
- `STELLA_PHOTOSPHERIC_MODEL_DIR`

Example:

```bash
export STELLA_INTERACTION_MODEL_DIR="$HOME/.cache/redback_surrogates/TypeII_Moriya/interaction_model"
export STELLA_PHOTOSPHERIC_MODEL_DIR="$HOME/.cache/redback_surrogates/TypeII_Moriya/photospheric_model"
```

## Download example

```bash
mkdir -p "$HOME/.cache/redback_surrogates/TypeII_Moriya/interaction_model"
mkdir -p "$HOME/.cache/redback_surrogates/TypeII_Moriya/photospheric_model"

curl -L \
  -o "$HOME/.cache/redback_surrogates/TypeII_Moriya/interaction_model/emulator_6param_timeweighted_best.pt" \
  "https://github.com/zheng-yang-zhang/redback_surrogates/releases/download/typeii-checkpoints-v1/emulator_6param_timeweighted_best.pt"

curl -L \
  -o "$HOME/.cache/redback_surrogates/TypeII_Moriya/photospheric_model/ae_cnn_v3_best.pt" \
  "https://github.com/zheng-yang-zhang/redback_surrogates/releases/download/typeii-checkpoints-v1/ae_cnn_v3_best.pt"

curl -L \
  -o "$HOME/.cache/redback_surrogates/TypeII_Moriya/photospheric_model/emulator_cnn_v3_6param_best.pt" \
  "https://github.com/zheng-yang-zhang/redback_surrogates/releases/download/typeii-checkpoints-v1/emulator_cnn_v3_6param_best.pt"
```

## Checksum verification

Copy `sha256sums.txt.template` to `sha256sums.txt` and run:

```bash
sha256sum -c sha256sums.txt
```

## Current scope

This checkpoint note covers only the two retained Type II surrogate pathways:

- interaction model
- photospheric model

The direct-regression checkpoint is intentionally excluded from this release plan.
