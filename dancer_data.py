"""
torch dataset implementation for loading phase chunks
"""

import math
import os
from bisect import bisect_right

import numpy as np
import soundfile
import torch

from torch.utils.data import Dataset

import config
import warnings


class OnlineWaveAugmenter:
    """
    Online, label-preserving waveform augmentation.
    Each transform rolls independently with its own probability.
    """
    def __init__(self, samplerate, config_dict, noise_files=None):
        self.sr = samplerate
        self.cfg = config_dict or {}
        self.noise_files = noise_files or []
        self._audiomentations = None
        self._build()

    def _build(self):
        try:
            import audiomentations as am
            self._audiomentations = am
        except Exception as e:
            self._audiomentations = None
            if self.cfg.get('enabled', False):
                warnings.warn(f'online augmentation requested but audiomentations import failed: {e}')
            return

        am = self._audiomentations
        self.t_gaussian_snr = None
        self.t_gaussian_noise = None
        self.t_color_noise = None
        self.t_room = None
        self.t_bg_noise = None
        self.t_short_noises = None

        try:
            self.t_gaussian_snr = am.AddGaussianSNR(p=1.0)
        except Exception:
            pass
        try:
            self.t_gaussian_noise = am.AddGaussianNoise(p=1.0)
        except Exception:
            pass
        try:
            self.t_color_noise = am.AddColorNoise(p=1.0)
        except Exception:
            pass
        try:
            self.t_room = am.RoomSimulator(p=1.0)
        except Exception:
            pass

        if self.noise_files:
            try:
                # Use folder root for recursive noise scan in audiomentations transforms.
                # If this fails due to version differences, we fall back silently.
                root = self.cfg.get('noise_corpus_path')
                if root:
                    self.t_bg_noise = am.AddBackgroundNoise(
                        sounds_path=root,
                        min_snr_db=self.cfg.get('background_noise_min_snr_db', 6.0),
                        max_snr_db=self.cfg.get('background_noise_max_snr_db', 24.0),
                        p=1.0,
                    )
            except Exception:
                self.t_bg_noise = None
            try:
                root = self.cfg.get('noise_corpus_path')
                if root:
                    self.t_short_noises = am.AddShortNoises(
                        sounds_path=root,
                        min_snr_db=self.cfg.get('short_noise_min_snr_db', 4.0),
                        max_snr_db=self.cfg.get('short_noise_max_snr_db', 24.0),
                        p=1.0,
                    )
            except Exception:
                self.t_short_noises = None

    def _roll(self, key, strength):
        p = float(self.cfg.get(key, 0.0)) * float(strength)
        if p <= 0:
            return False
        return np.random.random() < p

    def _time_mask(self, audio):
        max_seconds = float(self.cfg.get('max_mask_seconds', 8.0))
        max_len = int(max(0.0, max_seconds) * self.sr)
        if max_len <= 1 or len(audio) <= 1:
            return audio
        length = np.random.randint(1, min(max_len, len(audio)) + 1)
        start = np.random.randint(0, len(audio) - length + 1)
        out = audio.copy()
        out[start:start + length] = 0.0
        return out

    def _corpus_noise_patch(self, audio, strength):
        if not self.noise_files:
            return audio
        max_seconds = float(self.cfg.get('max_noise_seconds', 8.0))
        max_len = int(max(0.0, max_seconds) * self.sr)
        if max_len <= 1 or len(audio) <= 1:
            return audio

        nf = self.noise_files[np.random.randint(0, len(self.noise_files))]
        try:
            noise, nsr = soundfile.read(nf, dtype='float32')
        except Exception:
            return audio
        if noise.ndim == 2:
            noise = noise.mean(axis=1)
        if nsr != self.sr:
            # Keep this lightweight by avoiding resampling dependency in dataset path.
            return audio
        if len(noise) < 2:
            return audio

        length = np.random.randint(1, min(max_len, len(audio), len(noise)) + 1)
        n_start = np.random.randint(0, len(noise) - length + 1)
        a_start = np.random.randint(0, len(audio) - length + 1)

        noise_seg = noise[n_start:n_start + length]
        sig = audio[a_start:a_start + length]
        sig_rms = float(np.sqrt(np.mean(sig * sig)) + 1e-8)
        noise_rms = float(np.sqrt(np.mean(noise_seg * noise_seg)) + 1e-8)

        min_snr = float(self.cfg.get('patch_noise_min_snr_db', 6.0))
        max_snr = float(self.cfg.get('patch_noise_max_snr_db', 24.0))
        snr_db = np.random.uniform(min_snr, max_snr)
        snr_lin = 10.0 ** (snr_db / 20.0)
        target_noise_rms = sig_rms / snr_lin
        gain = target_noise_rms / noise_rms
        gain *= float(strength)

        out = audio.copy()
        out[a_start:a_start + length] = sig + noise_seg * gain
        return out

    def __call__(self, audio, strength=1.0):
        # audio: float32 np.ndarray mono
        out = audio.astype(np.float32, copy=True)

        if self._roll('p_time_mask', strength):
            out = self._time_mask(out)
        if self._roll('p_add_gaussian_snr', strength) and self.t_gaussian_snr is not None:
            out = self.t_gaussian_snr(samples=out, sample_rate=self.sr)
        if self._roll('p_add_gaussian_noise', strength) and self.t_gaussian_noise is not None:
            out = self.t_gaussian_noise(samples=out, sample_rate=self.sr)
        if self._roll('p_add_color_noise', strength) and self.t_color_noise is not None:
            out = self.t_color_noise(samples=out, sample_rate=self.sr)
        if self._roll('p_add_background_noise', strength):
            if self.t_bg_noise is not None:
                out = self.t_bg_noise(samples=out, sample_rate=self.sr)
            else:
                out = self._corpus_noise_patch(out, strength)
        if self._roll('p_add_short_noises', strength):
            if self.t_short_noises is not None:
                out = self.t_short_noises(samples=out, sample_rate=self.sr)
            else:
                out = self._corpus_noise_patch(out, strength)
        if self._roll('p_room_simulator', strength) and self.t_room is not None:
            out = self.t_room(samples=out, sample_rate=self.sr)

        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def load_bar_starts(bars_file_path):
    """Load bar start times (in seconds) from a .bars file."""
    values = []
    with open(bars_file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return values


def compute_labels(bar_starts, n_frames, frame_size, samplerate):
    """
    Compute phase and phase_rate labels from bar start times.

    Args:
        bar_starts: list of bar start times in seconds (relative to chunk start)
        n_frames:   number of output frames
        frame_size: samples per frame
        samplerate: audio sample rate

    Returns:
        phase_tensor: [n_frames] float32, phase in [0, 1)
        rate_tensor:  [n_frames] float32, phase_rate in radians/frame
                      (2π per bar, constant within each bar)
    """
    phase = np.zeros(n_frames, dtype=np.float32)
    rate  = np.zeros(n_frames, dtype=np.float32)

    if len(bar_starts) < 2:
        return torch.from_numpy(phase), torch.from_numpy(rate)

    bar_starts = sorted(bar_starts)

    for fi in range(n_frames):
        t = fi * frame_size / samplerate
        idx = bisect_right(bar_starts, t) - 1
        if idx < 0 or idx + 1 >= len(bar_starts):
            continue  # outside bar coverage — leave 0
        bar_start = bar_starts[idx]
        bar_end   = bar_starts[idx + 1]
        bar_dur   = bar_end - bar_start
        if bar_dur <= 0:
            continue
        phase[fi] = (t - bar_start) / bar_dur
        # rate: 2π radians per bar, expressed per frame
        rate[fi]  = 2.0 * math.pi * frame_size / (bar_dur * samplerate)

    return torch.from_numpy(phase), torch.from_numpy(rate)


class DanceDataset(Dataset):
    def __init__(self, data_path, frame_size, samplerate, augmenter=None, augment_ramp_epochs=0):
        self.path       = data_path
        self.frame_size = frame_size
        self.samplerate = samplerate
        self.augmenter = augmenter
        self.augment_ramp_epochs = int(augment_ramp_epochs)
        self.current_epoch = 0

        # collect (relative_stem, abs_dir) for every .ogg that has a matching .bars
        self.chunk_names = []  # list of (stem, directory) pairs
        for dirpath, _, files in os.walk(data_path):
            for f in files:
                if f.endswith('.ogg'):
                    stem = f[:-4]
                    ogg_path  = os.path.join(dirpath, f)
                    bars_path = os.path.join(dirpath, stem + '.bars')
                    if os.path.exists(bars_path) and os.path.getsize(ogg_path) > 0:
                        self.chunk_names.append((stem, dirpath))

    def __len__(self):
        return len(self.chunk_names)

    def set_epoch(self, epoch):
        self.current_epoch = max(0, int(epoch))

    def __getitem__(self, index):
        chunk_name, chunk_dir = self.chunk_names[index]

        audio_file = os.path.join(chunk_dir, chunk_name + '.ogg')
        bars_file  = os.path.join(chunk_dir, chunk_name + '.bars')

        audio, sr = soundfile.read(audio_file, dtype='float32')
        assert sr == self.samplerate, f'sample rate mismatch: {sr} != {self.samplerate}'
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        expected_frames  = int(config.chunk_duration * self.samplerate) // self.frame_size
        expected_samples = expected_frames * self.frame_size

        if audio.shape[0] < expected_samples:
            audio = np.pad(audio, (0, expected_samples - audio.shape[0])).astype(np.float32, copy=False)
        else:
            audio = audio[:expected_samples]

        if self.augmenter is not None:
            if self.augment_ramp_epochs > 0:
                strength = min(1.0, float(self.current_epoch + 1) / float(self.augment_ramp_epochs))
            else:
                strength = 1.0
            audio = self.augmenter(audio, strength=strength)

        bar_starts = load_bar_starts(bars_file)
        phase_labels, rate_labels = compute_labels(
            bar_starts, expected_frames, self.frame_size, self.samplerate)

        audio = torch.from_numpy(audio)
        frames = audio.reshape(expected_frames, self.frame_size)

        return frames, phase_labels, rate_labels, audio_file
