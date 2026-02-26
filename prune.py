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
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

import mido
from tqdm import tqdm

# Bump this string whenever filter logic changes — invalidates all existing markers.
PRUNE_VERSION = "3"

# ── MIDI thresholds ────────────────────────────────────────────────────────────
MIN_DURATION = 30       # seconds
MAX_DURATION = 60 * 20  # 20 minutes

# Valid SF2 time-signature denominators (powers of two, same as render_midi.py)
VALID_TS_DENOMINATORS = {1, 2, 4, 8, 16, 32}

# Fraction/grid alignment thresholds (phase-domain).
MAX_FRACTION_DENOMINATOR = 13
FRACTION_PHASE_TOLERANCE = 0.003
MIN_REGION_SCORE = 0.10
REGION_SECONDS = 16.0
MIN_NOTES_FOR_ALIGNMENT_CHECK = 24
MIN_NOTES_PER_REGION = 8

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


def _collect_unique_note_on_ticks(mid):
    ticks = set()
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                ticks.add(tick)
    return sorted(ticks)


def _collect_tempo_changes(mid):
    changes = {}
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'set_tempo':
                changes[tick] = msg.tempo
    tempos = [(t, changes[t]) for t in sorted(changes)]
    if not tempos or tempos[0][0] != 0:
        tempos = [(0, 500000)] + tempos
    return tempos


def _ticks_to_seconds(ticks, tempo_changes, ticks_per_beat):
    if not ticks:
        return []

    out = []
    ti = 0
    prev_tick = tempo_changes[0][0]
    acc = 0.0
    current_tempo = tempo_changes[0][1]

    for target in sorted(ticks):
        while ti + 1 < len(tempo_changes) and tempo_changes[ti + 1][0] <= target:
            next_tick, next_tempo = tempo_changes[ti + 1]
            acc += mido.tick2second(next_tick - prev_tick, ticks_per_beat, current_tempo)
            prev_tick = next_tick
            current_tempo = next_tempo
            ti += 1
        out.append(acc + mido.tick2second(target - prev_tick, ticks_per_beat, current_tempo))

    return out


def _phase_match_weight(phase):
    """
    Highest match weight among fractions k/d for d in [1..MAX_FRACTION_DENOMINATOR]
    when within FRACTION_PHASE_TOLERANCE.
    Weight formula: (13 + d) / d
    """
    best = 0.0
    for d in range(1, MAX_FRACTION_DENOMINATOR + 1):
        scaled = phase * d
        nearest_error = abs(scaled - round(scaled)) / d
        if nearest_error <= FRACTION_PHASE_TOLERANCE:
            w = (13.0 + d) / d
            if w > best:
                best = w
    return best


def _max_possible_weight():
    # d=1 yields the maximum weight for formula (13 + d) / d.
    return (13.0 + 1.0) / 1.0


def _has_good_note_phase_alignment(note_ticks, note_times, time_sigs, ticks_per_beat):
    """
    Returns (ok: bool, min_region_score: float, region_count: int, considered_notes: int).
    Each note contributes the weight 1/d if it lands near fraction k/d (best d chosen),
    with d in [1..MAX_FRACTION_DENOMINATOR], else 0.
    Regional score = normalized sum(weights) / notes_in_region, where each
    note weight is divided by max possible weight so score is in [0, 1].
    All sufficiently-populated regions must satisfy score >= MIN_REGION_SCORE.
    """
    if len(note_ticks) < MIN_NOTES_FOR_ALIGNMENT_CHECK:
        return False, 0.0, 0, len(note_ticks)

    tsi = 0
    region_total = {}
    region_score_sum = {}

    for tick, note_time in zip(note_ticks, note_times):
        while tsi + 1 < len(time_sigs) and time_sigs[tsi + 1][0] <= tick:
            tsi += 1

        ts_tick, num, den = time_sigs[tsi]
        bar_ticks = int(ticks_per_beat * 4 * num / den)
        if bar_ticks <= 0:
            continue

        phase = ((tick - ts_tick) % bar_ticks) / bar_ticks
        weight = _phase_match_weight(phase)

        rid = int(note_time / REGION_SECONDS) if REGION_SECONDS > 0 else 0
        region_total[rid] = region_total.get(rid, 0) + 1
        region_score_sum[rid] = region_score_sum.get(rid, 0.0) + weight

    considered_regions = []
    max_w = _max_possible_weight()
    for rid, total in region_total.items():
        if total >= MIN_NOTES_PER_REGION:
            score = (region_score_sum[rid] / max_w) / total
            considered_regions.append((rid, score, total))

    if not considered_regions:
        return False, 0.0, 0, len(note_ticks)

    min_score = min(score for _, score, _ in considered_regions)
    ok = min_score >= MIN_REGION_SCORE
    return ok, min_score, len(considered_regions), sum(total for _, _, total in considered_regions)


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

    note_ticks = _collect_unique_note_on_ticks(mid)
    tempo_changes = _collect_tempo_changes(mid)
    note_times = _ticks_to_seconds(note_ticks, tempo_changes, mid.ticks_per_beat)
    aligned_ok, min_region_score, region_count, aligned_count = _has_good_note_phase_alignment(
        note_ticks, note_times, time_sigs, mid.ticks_per_beat
    )
    if not aligned_ok:
        if aligned_count < MIN_NOTES_FOR_ALIGNMENT_CHECK:
            return False, (
                f"too few note-on ticks for alignment check "
                f"({aligned_count} < {MIN_NOTES_FOR_ALIGNMENT_CHECK})"
            )
        if region_count == 0:
            return False, (
                f"no region has enough notes for alignment check "
                f"(need >= {MIN_NOTES_PER_REGION} notes per {REGION_SECONDS:.0f}s region)"
            )
        return False, (
            f"poor fractional phase alignment "
            f"(min region score {min_region_score:.3f} < {MIN_REGION_SCORE:.3f}, "
            f"regions={region_count}, n={aligned_count})"
        )

    ts_label = ", ".join(f"{num}/{den}" for _, num, den in time_sigs)
    return True, (
        f"dur={duration:.0f}s ts={ts_label} "
        f"region_score>={MIN_REGION_SCORE:.2f} (regions={region_count}, n={aligned_count})"
    )


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

def process_midi(
    midi_path: Path,
    delete: bool,
    workers: int,
    fast: bool,
    use_markers: bool,
    move_kept_to: Path | None = None,
    overwrite_out: bool = False,
):
    files = sorted(midi_path.rglob('*.mid'))
    total = len(files)
    mode  = 'fast' if fast else 'mido'
    print(f"\n{'='*60}")
    print(f"MIDI: scanning {total} files in {midi_path}  (workers={workers}, mode={mode}, markers={'on' if use_markers else 'off'})")
    print(f"{'='*60}")

    kept = 0
    pruned = 0
    skipped = 0
    moved = 0
    move_errors = 0
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
                if keep and move_kept_to is not None:
                    rel = f.relative_to(midi_path)
                    dest = move_kept_to / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and not overwrite_out:
                        move_errors += 1
                        if not delete:
                            print(f"  SKIP MOVE {rel}  — destination exists")
                    else:
                        try:
                            if dest.exists() and overwrite_out:
                                dest.unlink()
                            shutil.move(str(f), str(dest))
                            moved += 1
                            # Move marker alongside file if present.
                            marker_src = _marker_path(f)
                            if marker_src.exists():
                                marker_dest = dest.with_suffix('.mid.prune_ok')
                                marker_dest.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(marker_src), str(marker_dest))
                        except Exception as e:
                            move_errors += 1
                            if not delete:
                                print(f"  MOVE ERR {rel}  — {e}")

                bar.set_postfix(kept=kept, pruned=pruned, moved=moved, skip=skipped, refresh=False)
                bar.update(1)

    for line in prune_lines:
        print(line)

    print(
        f"\nMIDI summary: kept={kept}  pruned={pruned}  moved={moved}  "
        f"move_errors={move_errors}  skipped(marker)={skipped}  total={total}"
    )
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
    global MAX_FRACTION_DENOMINATOR
    global FRACTION_PHASE_TOLERANCE
    global MIN_REGION_SCORE
    global REGION_SECONDS
    global MIN_NOTES_FOR_ALIGNMENT_CHECK
    global MIN_NOTES_PER_REGION

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
    parser.add_argument('--move-kept-to', type=Path, default=None,
                        help='Move kept MIDI files to this output directory (preserving relative paths)')
    parser.add_argument('--overwrite-out', action='store_true',
                        help='When used with --move-kept-to, overwrite existing files in output')
    parser.add_argument('--max-fraction-denominator', type=int, default=MAX_FRACTION_DENOMINATOR,
                        help=f'Max denominator for phase fractions k/d (default: {MAX_FRACTION_DENOMINATOR})')
    parser.add_argument('--fraction-phase-tolerance', type=float, default=FRACTION_PHASE_TOLERANCE,
                        help=f'Max phase error to match fraction (default: {FRACTION_PHASE_TOLERANCE})')
    parser.add_argument('--min-region-score', type=float, default=MIN_REGION_SCORE,
                        help=f'Min weighted score per region (default: {MIN_REGION_SCORE})')
    parser.add_argument('--region-seconds', type=float, default=REGION_SECONDS,
                        help=f'Region length in seconds for local alignment checks (default: {REGION_SECONDS})')
    parser.add_argument('--min-notes-for-alignment-check', type=int, default=MIN_NOTES_FOR_ALIGNMENT_CHECK,
                        help=f'Min unique note-on ticks for alignment check (default: {MIN_NOTES_FOR_ALIGNMENT_CHECK})')
    parser.add_argument('--min-notes-per-region', type=int, default=MIN_NOTES_PER_REGION,
                        help=f'Min notes in region for regional score check (default: {MIN_NOTES_PER_REGION})')
    args = parser.parse_args()

    if args.midi_path is None and args.sf_path is None:
        parser.error('Provide at least one of --midi-path or --sf-path')

    move_mode = args.move_kept_to is not None

    if not args.delete and not move_mode:
        print("DRY RUN — pass --delete to actually remove files")
    if move_mode:
        print(f"MOVE MODE — kept MIDI files will be moved to {args.move_kept_to}")

    MAX_FRACTION_DENOMINATOR = args.max_fraction_denominator
    FRACTION_PHASE_TOLERANCE = args.fraction_phase_tolerance
    MIN_REGION_SCORE = args.min_region_score
    REGION_SECONDS = args.region_seconds
    MIN_NOTES_FOR_ALIGNMENT_CHECK = args.min_notes_for_alignment_check
    MIN_NOTES_PER_REGION = args.min_notes_per_region

    # In move-mode we must rescan every file in source and never skip by marker.
    use_markers = (not args.no_markers) and (not move_mode)
    if use_markers:
        print(f"Markers enabled (version={PRUNE_VERSION}) — already-passed files will be skipped")
    elif move_mode:
        print("Markers disabled in move-mode (no file skipping).")

    if args.midi_path:
        process_midi(
            args.midi_path,
            args.delete,
            args.workers,
            args.fast,
            use_markers,
            move_kept_to=args.move_kept_to,
            overwrite_out=args.overwrite_out,
        )
    if args.sf_path:
        process_soundfonts(args.sf_path, args.delete, args.min_gm_families)


if __name__ == '__main__':
    main()
