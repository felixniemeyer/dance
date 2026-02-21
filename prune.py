"""
Prune MIDI files and soundfonts that are unsuitable for training.

MIDI filters applied:
  - Parse errors
  - Duration too short (< MIN_DURATION) or too long (> MAX_DURATION)
  - Missing or invalid time-signature data
    (must have a time-sig at tick 0; denominator must be a power-of-two ≤ 32)

Soundfont filters applied:
  - File too small (< MIN_SF2_SIZE bytes)
  - Not a valid SF2 file
  - Insufficient GM instrument family coverage in bank 0
    (must cover at least MIN_GM_FAMILIES of the 16 GM families)
  - No drum bank (SF2 bank=128) in preset table

Usage:
  # Dry run - just report, don't delete:
  python prune.py --midi-path midi_files/lmd_matched --sf-path data

  # Actually delete:
  python prune.py --midi-path midi_files/lmd_matched --sf-path data --delete

  # Only MIDI (parallel):
  python prune.py --midi-path midi_files/lmd_matched --workers 8

  # Only soundfonts:
  python prune.py --sf-path data
"""

import os
import sys
import struct
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import mido
from tqdm import tqdm

# Bump this string whenever filter logic changes — invalidates all existing markers.
PRUNE_VERSION = "1"

# ── MIDI thresholds ────────────────────────────────────────────────────────────
MIN_DURATION = 30       # seconds
MAX_DURATION = 60 * 20  # 20 minutes

# Valid SF2 time-signature denominators (powers of two, same as render_midi.py)
VALID_TS_DENOMINATORS = {1, 2, 4, 8, 16, 32}

# ── Soundfont thresholds ───────────────────────────────────────────────────────
MIN_SF2_SIZE    = 200_000  # bytes
MIN_GM_FAMILIES = 8        # of 16; require coverage across at least this many families

# GM bank-0 families: programs 0-127 split into 16 families of 8 programs each.
GM_FAMILY_NAMES = [
    "Piano", "Chromatic Perc", "Organ", "Guitar",
    "Bass", "Strings", "Ensemble", "Brass",
    "Reed", "Pipe", "Synth Lead", "Synth Pad",
    "Synth Effects", "Ethnic", "Percussive", "Sound Effects",
]


# ── MIDI filtering ─────────────────────────────────────────────────────────────

def _has_valid_time_signature(time_sigs):
    """
    Mirror of render_midi.py has_valid_time_signature(), operating on a list of
    (tick, numerator, denominator) tuples.
    """
    if not time_sigs:
        return False
    if time_sigs[0][0] != 0:
        return False
    for _, num, den in time_sigs:
        if num <= 0:
            return False
        if den not in VALID_TS_DENOMINATORS:
            return False
    return True


def check_midi(path: Path):
    """
    Returns (keep: bool, reason: str).
    """
    try:
        mid = mido.MidiFile(str(path))
    except Exception as e:
        return False, f"parse error: {e}"

    try:
        duration = mid.length
    except Exception as e:
        return False, f"parse error: {e}"
    if duration < MIN_DURATION:
        return False, f"too short ({duration:.0f}s < {MIN_DURATION}s)"
    if duration > MAX_DURATION:
        return False, f"too long ({duration:.0f}s > {MAX_DURATION}s)"

    # Collect time-signature events across all tracks, deduplicated by tick.
    ts_by_tick = {}
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'time_signature':
                ts_by_tick[tick] = (tick, msg.numerator, msg.denominator)

    time_sigs = [ts_by_tick[t] for t in sorted(ts_by_tick)]

    if not _has_valid_time_signature(time_sigs):
        if not time_sigs:
            return False, "no time-signature events"
        first_tick = time_sigs[0][0]
        if first_tick != 0:
            return False, f"first time-sig not at tick 0 (tick={first_tick})"
        bad = [(num, den) for _, num, den in time_sigs if den not in VALID_TS_DENOMINATORS or num <= 0]
        return False, f"invalid time-signature: {bad[0] if bad else '?'}"

    ts_label = ", ".join(f"{num}/{den}" for _, num, den in time_sigs)
    return True, f"dur={duration:.0f}s ts={ts_label}"


# ── Marker helpers ─────────────────────────────────────────────────────────────

def _marker_path(midi_path: Path) -> Path:
    return midi_path.with_suffix('.mid.prune_ok')

def _has_valid_marker(midi_path: Path) -> bool:
    try:
        return _marker_path(midi_path).read_text().strip() == PRUNE_VERSION
    except OSError:
        return False

def _write_marker(midi_path: Path):
    try:
        _marker_path(midi_path).write_text(PRUNE_VERSION)
    except OSError:
        pass


# ── Fast binary MIDI check (no mido) ──────────────────────────────────────────

def check_midi_fast(path: Path):
    """
    Lightweight binary scan — no mido, no Python objects per event.
    Checks: magic bytes, format type, SMPTE flag, time-signature presence.
    Does NOT check duration (requires full tempo parse) or tick-0 placement.
    Faster than check_midi but slightly less strict — use as a first pass.
    Returns (keep: bool, reason: str).
    """
    try:
        size = path.stat().st_size
        if size < 200:
            return False, "too small"

        with open(path, 'rb') as f:
            hdr = f.read(14)

        if len(hdr) < 14 or hdr[:4] != b'MThd':
            return False, "not a MIDI file (bad magic)"

        fmt      = struct.unpack('>H', hdr[8:10])[0]
        division = struct.unpack('>H', hdr[12:14])[0]

        if fmt == 2:
            return False, "type 2 (asynchronous) MIDI"
        if division & 0x8000:
            return False, "SMPTE timecode (not tempo-based)"

        # Scan raw bytes for time-signature meta event marker: FF 58 04
        data = path.read_bytes()
        if b'\xff\x58\x04' not in data:
            return False, "no time-signature events"

        return True, "fast-pass ok"
    except Exception as e:
        return False, f"parse error: {e}"


def _check_midi_worker(args):
    path, delete, fast, use_markers = args

    if use_markers and _has_valid_marker(path):
        return path, True, 'marker'

    check_fn = check_midi_fast if fast else check_midi
    keep, reason = check_fn(path)

    if keep and use_markers:
        _write_marker(path)
    elif not keep and delete:
        try:
            path.unlink()
            _marker_path(path).unlink(missing_ok=True)
        except OSError:
            pass
    return path, keep, reason


# ── SF2 filtering ──────────────────────────────────────────────────────────────

def _parse_sf2_presets(path: Path):
    """
    Return list of (bank, preset_num, name) or None on parse failure.
    Reads only the pdta/phdr sub-chunk (preset headers) — fast even for large files.
    """
    try:
        with open(path, 'rb') as f:
            if f.read(4) != b'RIFF':
                return None
            f.read(4)  # chunk size
            if f.read(4) != b'sfbk':
                return None

            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size = struct.unpack('<I', f.read(4))[0]

                if chunk_id == b'LIST':
                    list_type = f.read(4)
                    if list_type == b'pdta':
                        remaining = chunk_size - 4
                        presets = []
                        while remaining > 0:
                            sub_id   = f.read(4)
                            sub_size = struct.unpack('<I', f.read(4))[0]
                            remaining -= 8 + sub_size
                            if sub_id == b'phdr':
                                for _ in range(sub_size // 38):
                                    data = f.read(38)
                                    if len(data) < 38:
                                        break
                                    name   = data[:20].rstrip(b'\x00').decode('ascii', errors='replace')
                                    preset = struct.unpack('<H', data[20:22])[0]
                                    bank   = struct.unpack('<H', data[22:24])[0]
                                    presets.append((bank, preset, name))
                            else:
                                f.seek(sub_size, 1)
                        return presets
                    else:
                        f.seek(chunk_size - 4, 1)
                else:
                    f.seek(chunk_size, 1)
    except Exception:
        return None
    return []


def check_soundfont(path: Path, min_gm_families: int = MIN_GM_FAMILIES):
    """
    Returns (keep: bool, reason: str).
    """
    size = path.stat().st_size
    if size < MIN_SF2_SIZE:
        return False, f"too small ({size / 1024:.0f} KB)"

    presets = _parse_sf2_presets(path)
    if presets is None:
        return False, "SF2 parse error / not a valid SF2"

    banks = {p[0] for p in presets}

    if 128 not in banks:
        return False, f"no drum bank (banks present: {sorted(banks)[:8]})"

    # Check GM family coverage in bank 0.
    bank0_programs = {p[1] for p in presets if p[0] == 0 and p[1] < 128}
    covered_families = {prog // 8 for prog in bank0_programs}
    n_families = len(covered_families)

    if n_families < min_gm_families:
        missing = [GM_FAMILY_NAMES[i] for i in range(16) if i not in covered_families]
        return False, (
            f"insufficient GM coverage: {n_families}/{len(GM_FAMILY_NAMES)} families "
            f"(missing: {', '.join(missing[:4])}{'…' if len(missing) > 4 else ''})"
        )

    return True, f"ok — {len(presets)} presets, {n_families}/16 GM families, banks={sorted(banks)[:8]}"


# ── Main ───────────────────────────────────────────────────────────────────────

def process_midi(midi_path: Path, delete: bool, workers: int, fast: bool, use_markers: bool):
    files = sorted(midi_path.rglob('*.mid'))
    total = len(files)
    mode  = 'fast' if fast else 'mido'
    print(f"\n{'='*60}")
    print(f"MIDI: scanning {total} files in {midi_path}  (workers={workers}, mode={mode}, markers={'on' if use_markers else 'off'})")
    print(f"{'='*60}")

    kept    = 0
    pruned  = 0
    skipped = 0
    reasons = {}

    args_iter  = ((f, delete, fast, use_markers) for f in files)
    prune_lines = []

    with Pool(processes=workers) as pool:
        with tqdm(total=total, unit='file', dynamic_ncols=True) as bar:
            for f, keep, reason in pool.imap_unordered(_check_midi_worker, args_iter, chunksize=64):
                if reason == 'marker':
                    skipped += 1
                    kept += 1
                elif keep:
                    kept += 1
                else:
                    pruned += 1
                    tag = reason.split('(')[0].split(':')[0].strip()
                    reasons[tag] = reasons.get(tag, 0) + 1
                    if not delete:
                        prune_lines.append(f"  PRUNE  {f.relative_to(midi_path)}  — {reason}")
                bar.set_postfix(kept=kept, pruned=pruned, skip=skipped, refresh=False)
                bar.update(1)

    for line in prune_lines:
        print(line)

    print(f"\nMIDI summary: kept={kept}  pruned={pruned}  skipped(marker)={skipped}  total={total}")
    if reasons:
        print("  Prune reasons:")
        for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {n:6d}  {r}")


def process_soundfonts(sf_path: Path, delete: bool, min_gm_families: int):
    files = sorted(sf_path.glob('*.sf2'))
    total = len(files)
    print(f"\n{'='*60}")
    print(f"Soundfonts: scanning {total} files in {sf_path}")
    print(f"{'='*60}")

    kept    = 0
    pruned  = 0
    reasons = {}

    for f in files:
        keep, reason = check_soundfont(f, min_gm_families=min_gm_families)
        if keep:
            kept += 1
            print(f"  KEEP   {f.name[:70]}")
        else:
            pruned += 1
            tag = reason.split('(')[0].split('/')[0].strip()
            reasons[tag] = reasons.get(tag, 0) + 1
            print(f"  PRUNE  {f.name[:60]}  — {reason}")
            if delete:
                f.unlink()

    print(f"\nSoundfonts summary: kept={kept}  pruned={pruned}  total={total}")
    if reasons:
        print("  Prune reasons:")
        for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {n:6d}  {r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--midi-path', type=Path, default=None,
                        help='Root directory of MIDI files (searched recursively)')
    parser.add_argument('--sf-path', type=Path, default=None,
                        help='Directory containing .sf2 soundfont files')
    parser.add_argument('--delete', action='store_true',
                        help='Actually delete pruned files (default: dry run)')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1),
                        help='Parallel worker processes for MIDI scanning (default: cpu_count-1)')
    parser.add_argument('--min-gm-families', type=int, default=MIN_GM_FAMILIES,
                        help=f'Min GM instrument families required in bank 0 (default: {MIN_GM_FAMILIES}/16)')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast binary scanner instead of mido (skips duration check, less strict)')
    parser.add_argument('--no-markers', action='store_true',
                        help='Disable marker files (always re-scan every file)')
    args = parser.parse_args()

    if args.midi_path is None and args.sf_path is None:
        parser.error('Provide at least one of --midi-path or --sf-path')

    if not args.delete:
        print("DRY RUN — pass --delete to actually remove files")

    use_markers = not args.no_markers
    if use_markers:
        print(f"Markers enabled (version={PRUNE_VERSION}) — already-passed files will be skipped")

    if args.midi_path:
        process_midi(args.midi_path, args.delete, args.workers, args.fast, use_markers)
    if args.sf_path:
        process_soundfonts(args.sf_path, args.delete, args.min_gm_families)


if __name__ == '__main__':
    main()
