# Training Guide (Phase Model)

This project trains for **bar phase estimation** (scalar in `[0, 1)` representing position within the current bar).

## Architecture

Active model: `phase_gru_mel` — GRU with sliding-window mel spectrogram frontend.

Outputs `[sin(phase), cos(phase), phase_rate]` per frame:
- `sin/cos`: current bar phase encoded as a unit vector
- `phase_rate`: radians per frame (implicitly learned; use for lookahead at inference time)

## Config

- samplerate: `16000`
- chunk duration: `32s`
- frame size: `320` (50 FPS)

## Training

```bash
python train.py phase_gru_mel \
  --chunks-path chunks/fv1_v0 \
  -t <tag> -e <epochs> -b <batch_size> \
  -r 1e-4 -rd 0.977
```

Continue from checkpoint:

```bash
python train.py phase_gru_mel \
  --chunks-path chunks/fv1_v0 \
  -t <tag> --continue-from <epoch> \
  -e <more_epochs> -r 1e-5 -b 64
```

## Overfit sanity check

```bash
python train.py phase_gru_mel --chunks-path chunks/fv1_v0 \
  -t overfit_test -e 50 -b 4 --dataset-size 4
```

Training loss should fall clearly within ~50 epochs.

## Inference

```bash
python apply.py checkpoints/<tag>/<epoch>.pt <audio.ogg>
```

## Experiment tracking

```bash
mlflow server --host 0.0.0.0 --port 5000
```
