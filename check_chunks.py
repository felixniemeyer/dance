"""
Evaluate a trained model on every chunk individually and report per-chunk loss.
Useful for finding bad labels, corrupt audio, or outliers in the dataset.

Usage:
    python check_chunks.py checkpoints/gru_mel_no_anticipation/100.pt chunks/fv1_v0
    python check_chunks.py checkpoints/gru_mel_no_anticipation/100.pt chunks/fv1_v0 --top 30
    python check_chunks.py checkpoints/gru_mel_no_anticipation/100.pt chunks/fv1_v0 --csv out.csv
"""

import argparse
import os
import csv

import torch
import soundfile
import numpy as np

import config
from models.selector import loadModel
from dancer_data import load_phase_labels

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str)
parser.add_argument('chunks_path', type=str)
parser.add_argument('--top', type=int, default=20, help='print N worst chunks')
parser.add_argument('--csv', type=str, default=None, help='save full results to CSV')
parser.add_argument('--warmup-seconds', type=float, default=8.0)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = loadModel(args.checkpoint)
model.to(device)
model.eval()

frame_size    = config.frame_size
samplerate    = config.samplerate
expected_frames = int(config.chunk_duration * samplerate) // frame_size
expected_samples = expected_frames * frame_size
warmup_frames = int(args.warmup_seconds / (frame_size / samplerate))

chunk_names = [
    f[:-4] for f in os.listdir(args.chunks_path)
    if f.endswith('.ogg')
    and os.path.exists(os.path.join(args.chunks_path, f[:-4] + '.phase'))
]
print(f'{len(chunk_names)} chunks found')

results = []

with torch.no_grad():
    for i, name in enumerate(chunk_names):
        print(f'\r{i+1}/{len(chunk_names)}  ', end='', flush=True)

        audio_path = os.path.join(args.chunks_path, name + '.ogg')
        phase_path = os.path.join(args.chunks_path, name + '.phase')

        try:
            audio, sr = soundfile.read(audio_path, dtype='float32')
            assert sr == samplerate, f'samplerate mismatch: {sr}'
        except Exception as e:
            results.append((999.0, name, f'audio error: {e}'))
            continue

        try:
            phase_labels = load_phase_labels(phase_path)
        except Exception as e:
            results.append((999.0, name, f'label error: {e}'))
            continue

        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = torch.from_numpy(audio)
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        if audio.shape[0] < expected_samples:
            audio = torch.nn.functional.pad(audio, (0, expected_samples - audio.shape[0]))
        else:
            audio = audio[:expected_samples]

        if phase_labels.shape[0] < expected_frames:
            phase_labels = torch.nn.functional.pad(phase_labels, (0, expected_frames - phase_labels.shape[0]))

        frames = audio.reshape(1, expected_frames, frame_size).to(device)
        phase_labels = phase_labels[:expected_frames].unsqueeze(0).to(device)

        outputs, _ = model(frames)
        preds = outputs[:, warmup_frames:, :2]
        labels = phase_labels[:, warmup_frames:]

        target_angles  = labels * 2 * torch.pi
        target_vectors = torch.stack([torch.sin(target_angles), torch.cos(target_angles)], dim=-1)
        pred_norm      = preds.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        normalized     = preds / pred_norm
        loss           = (normalized - target_vectors).pow(2).mean().item()

        results.append((loss, name, ''))

print()
results.sort(key=lambda x: x[0], reverse=True)

print(f'\n{"Loss":>8}  Chunk')
print('-' * 60)
for loss, name, note in results[:args.top]:
    flag = f'  ← {note}' if note else ''
    print(f'{loss:8.4f}  {name}{flag}')

# summary stats
losses = [r[0] for r in results if r[0] < 999.0]
losses_arr = np.array(losses)
print(f'\nTotal: {len(losses)} chunks evaluated')
print(f'Mean:   {losses_arr.mean():.4f}')
print(f'Median: {np.median(losses_arr):.4f}')
print(f'p90:    {np.percentile(losses_arr, 90):.4f}')
print(f'p99:    {np.percentile(losses_arr, 99):.4f}')
print(f'Max:    {losses_arr.max():.4f}')

# histogram buckets
print('\nDistribution:')
buckets = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 2.0]
for lo, hi in zip(buckets, buckets[1:]):
    count = ((losses_arr >= lo) & (losses_arr < hi)).sum()
    bar   = '█' * (count * 40 // max(1, len(losses)))
    print(f'  [{lo:.1f}, {hi:.1f})  {bar} {count}')

if args.csv:
    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['loss', 'chunk', 'note'])
        w.writerows(results)
    print(f'\nFull results saved to {args.csv}')
