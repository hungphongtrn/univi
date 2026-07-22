"""Temporal attention-alignment test for spectrogram reading.

Question: when the model emits the token for a word spoken at time T, does its
attention over the spectrogram land on the image column corresponding to T?
A time-aligned diagonal that (a) matches Whisper's *non-uniform* word timestamps
and (b) collapses under a permuted (wrong) spectrogram is strong evidence the
model literally reads the pixels, not autoregressive positional habit.

Layout (confirmed by feasibility.py): a 10 s page = 246 contiguous image soft
tokens = 6 freq rows x 41 time cols, row-major. Column c spans [c/41*10, ...] s.

Controls:
  aligned   : real spectrogram
  permuted  : another clip's spectrogram, SAME target text + token positions
              -> a diagonal here would be positional habit, not reading
  expected column comes from Whisper word timestamps (non-uniform), so tracking
  it beats tracking a uniform token-position ramp.

Reuses eval_lane seams + wer_direct's local-LibriSpeech loader + production render.
Run:
  uv run python data/eval/attn_probe/temporal_align.py --n-clips 24 \
      --checkpoint data/checkpoints/audio-only-v0/checkpoint-800
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import numpy as np

from eval_lane import _collate_one, _merge_images, load_model
from data.preprocessing.render_utils import render_log_mel_spectrogram

# --- decodability dir shares the local librispeech + whisper ---
DECOD = Path(__file__).resolve().parents[1] / "decodability"
LIBRI_DIR = DECOD / "LibriSpeech" / "test-clean"
WHISPER_ID = str(DECOD / "whisper-tiny.en")

SR = 16000
PAGE_DUR = 10.0
N_COLS = 41           # time columns in the 246-token grid
N_ROWS = 6            # freq rows
IMG_TOKEN_ID = 258880
PROMPT = "Transcribe the speech represented by this spectrogram image."


# --------------------------------------------------------------------------- #
# Data: local LibriSpeech FLACs -> (audio, ref) -> rendered page + messages
# --------------------------------------------------------------------------- #

def load_clips(n_clips: int, max_dur: float = 10.0):
    import soundfile as sf
    import librosa

    transcripts: dict[str, str] = {}
    for tf in sorted(LIBRI_DIR.rglob("*.trans.txt")):
        for line in tf.read_text().splitlines():
            uid, _, text = line.partition(" ")
            transcripts[uid] = text
    clips = []
    for flac in sorted(LIBRI_DIR.rglob("*.flac")):
        audio, sr = sf.read(str(flac), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        dur = len(audio) / SR
        if dur > max_dur or dur < 2.0:   # single page, and long enough to have structure
            continue
        ref = transcripts.get(flac.stem)
        if not ref:
            continue
        clips.append({"audio": audio, "ref": ref, "dur": dur, "uid": flac.stem})
        if len(clips) >= n_clips:
            break
    return clips


def render_page(audio: np.ndarray):
    pages = render_log_mel_spectrogram(
        audio, sample_rate=SR, source_sample_rate=SR, n_mels=80, n_fft=400,
        hop_length=160, window="hann", power=2.0, page_duration_sec=PAGE_DUR,
        max_pages=1, output_width=1000, output_height=160,
        normalization="whisper_log_mel",
    )
    return pages[0] if isinstance(pages, list) else pages


def build_messages(ref: str):
    return [
        {"role": "user", "content": [
            {"type": "image", "text": None},
            {"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": ref}]},
    ]


# --------------------------------------------------------------------------- #
# Whisper word timestamps (local tiny.en)
# --------------------------------------------------------------------------- #

def whisper_word_times(clips):
    """Return, per clip, a list of (word_lower, mid_time_sec)."""
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=WHISPER_ID,
                   device=device, return_timestamps="word")
    out = []
    for c in clips:
        res = asr(c["audio"].copy(), return_timestamps="word")
        words = []
        for ch in res.get("chunks", []):
            ts = ch.get("timestamp") or (None, None)
            s, e = ts
            if s is None:
                continue
            if e is None:
                e = s
            w = ch["text"].strip().lower()
            w = "".join(ch2 for ch2 in w if ch2.isalnum())
            if w:
                words.append((w, 0.5 * (s + e)))
        out.append(words)
    del asr
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Token -> word index mapping for the response (assistant) tokens
# --------------------------------------------------------------------------- #

def response_token_words(tokenizer, input_ids, labels):
    """For each response token (label != -100), return its 0-based word index in
    the emitted transcript. Word boundaries are detected from the sub-word
    tokens (a new word starts on a space/▁ prefix)."""
    import torch

    tok = getattr(tokenizer, "tokenizer", tokenizer)  # Gemma4Processor wraps the tokenizer
    resp_pos = torch.where(labels != -100)[0]
    toks = tok.convert_ids_to_tokens([int(input_ids[p]) for p in resp_pos])
    word_idx = []
    w = -1
    for t in toks:
        starts_word = t.startswith("▁") or t.startswith("Ġ") or t.startswith(" ")
        # first real token also starts a word
        if starts_word or w == -1:
            w += 1
        word_idx.append(w)
    return resp_pos, word_idx


# --------------------------------------------------------------------------- #
# Attention -> per-token time-column centroid
# --------------------------------------------------------------------------- #

def image_block_positions(input_ids):
    import torch
    pos = torch.where(input_ids == IMG_TOKEN_ID)[0]
    return pos


def per_layer_centroids(attns, resp_pos, img_pos, transpose=False):
    """Return array (n_layers, n_resp) of time-column centroids in [0, N_COLS).

    attns[l]: (1, heads, q, k). Average over heads. For each response query,
    take attention over the 246 image keys, reshape (6,41) row-major, sum freq
    rows -> 41-col profile, centroid. transpose=True reshapes (41,6) instead
    (orientation check)."""
    import torch

    n_layers = len(attns)
    out = np.full((n_layers, len(resp_pos)), np.nan, dtype=np.float64)
    cols = np.arange(N_COLS, dtype=np.float64)
    for l, a in enumerate(attns):
        am = a[0].float().mean(0)              # (q, k)
        block = am[resp_pos][:, img_pos]       # (n_resp, 246)
        block = block.cpu().numpy()
        for i in range(block.shape[0]):
            v = block[i]
            if v.sum() <= 0:
                continue
            grid = v.reshape(N_COLS, N_ROWS) if transpose else v.reshape(N_ROWS, N_COLS)
            prof = grid.sum(axis=0) if not transpose else grid.sum(axis=1)  # (41,)
            s = prof.sum()
            if s <= 0:
                continue
            out[l, i] = float((cols * prof).sum() / s)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def spearman(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="data/checkpoints/audio-only-v0/checkpoint-800")
    ap.add_argument("--n-clips", type=int, default=24)
    ap.add_argument("--transpose", action="store_true", help="reshape (41,6) instead of (6,41)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    print(f"[data] loading up to {args.n_clips} librispeech clips ...", flush=True)
    clips = load_clips(args.n_clips)
    print(f"[data] {len(clips)} clips; durations {min(c['dur'] for c in clips):.1f}-{max(c['dur'] for c in clips):.1f}s", flush=True)

    print("[whisper] word timestamps ...", flush=True)
    word_times = whisper_word_times(clips)
    for c, wt in zip(clips[:2], word_times[:2]):
        print(f"   {c['uid']}: {len(wt)} words; first={wt[:4]}", flush=True)

    # render pages once; permutation donor = next clip's page
    pages = [render_page(c["audio"]) for c in clips]

    print(f"[model] loading {args.checkpoint} ...", flush=True)
    model, tokenizer = load_model(args.checkpoint)
    model.eval()
    for cfg in (model.config, getattr(model.config, "text_config", None),
                getattr(model.config, "vision_config", None)):
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"

    # per-clip, per-layer within-clip Spearman(expected, predicted) for each lane.
    n_layers = None
    clip_corr = {"aligned": [], "permuted": []}   # list over clips of array(n_layers)
    pooled = {"aligned": None, "permuted": None}   # pooled (secondary)

    for ci, c in enumerate(clips):
        wt = word_times[ci]
        if len(wt) < 3:
            continue
        for lane in ("aligned", "permuted"):
            img = pages[ci] if lane == "aligned" else pages[(ci + 1) % len(pages)]
            msgs = build_messages(c["ref"])
            batch = _collate_one(model, tokenizer, _merge_images(msgs, [img]))
            input_ids = batch["input_ids"][0]
            labels = batch["labels"][0]
            img_pos = image_block_positions(input_ids)
            if len(img_pos) != N_COLS * N_ROWS:
                print(f"   [warn] clip {ci} {lane}: {len(img_pos)} image tokens (!=246), skipping", flush=True)
                continue
            resp_pos, word_idx = response_token_words(tokenizer, input_ids, labels)
            with torch.no_grad():
                out = model(**batch, output_attentions=True)
            attns = out.attentions
            if n_layers is None:
                n_layers = len(attns)
                pooled["aligned"] = [([], []) for _ in range(n_layers)]
                pooled["permuted"] = [([], []) for _ in range(n_layers)]
            cents = per_layer_centroids(attns, resp_pos, img_pos, transpose=args.transpose)
            n_words = len(wt)
            exp_col = np.array(
                [min(max(wt[wi][1] / PAGE_DUR * N_COLS, 0.0), N_COLS - 1) if wi < n_words else np.nan
                 for wi in word_idx], dtype=np.float64)
            per_layer_c = np.array([spearman(exp_col, cents[l]) for l in range(n_layers)])
            clip_corr[lane].append(per_layer_c)
            for l in range(n_layers):
                pooled[lane][l][0].extend(exp_col.tolist())
                pooled[lane][l][1].extend(cents[l].tolist())
        print(f"   [{ci+1}/{len(clips)}] {c['uid']} done", flush=True)

    def mean_over_clips(lane):
        arr = np.array(clip_corr[lane])  # (n_used_clips, n_layers)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), arr.shape[0]

    al_m, al_s, n_used = mean_over_clips("aligned")
    pm_m, pm_s, _ = mean_over_clips("permuted")

    report = {"checkpoint": args.checkpoint, "n_clips": len(clips), "n_used": int(n_used),
              "transpose": args.transpose, "metric": "within-clip Spearman(whisper_expected_col, predicted_col), mean over clips",
              "layers": []}
    print("\n" + "=" * 78, flush=True)
    print("TEMPORAL ALIGNMENT  within-clip Spearman(whisper expected col, predicted col)", flush=True)
    print(f"  mean over {n_used} clips.  aligned>0 & permuted~0 => reads the time axis", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'layer':>5} {'aligned':>9} {'permuted':>9} {'Δ(al-perm)':>11}", flush=True)
    for l in range(n_layers or 0):
        sa, sp = float(al_m[l]), float(pm_m[l])
        d = sa - sp
        report["layers"].append({"layer": l, "aligned": sa, "permuted": sp, "delta": d,
                                 "aligned_std": float(al_s[l]), "permuted_std": float(pm_s[l]),
                                 "pooled_aligned": spearman(*pooled["aligned"][l]),
                                 "pooled_permuted": spearman(*pooled["permuted"][l])})
        star = " *" if d > 0.15 else ""
        print(f"  {l:>5} {sa:>9.3f} {sp:>9.3f} {d:>11.3f}{star}", flush=True)

    # deep-band summary (layers 28..last) — where reading concentrated in the pilot
    band = [l for l in range(n_layers) if l >= 28]
    db_al = float(np.nanmean([al_m[l] for l in band]))
    db_pm = float(np.nanmean([pm_m[l] for l in band]))
    best_l = int(np.nanargmax([report["layers"][l]["delta"] for l in range(n_layers)]))
    report["deep_band"] = {"layers": band, "aligned": db_al, "permuted": db_pm, "delta": db_al - db_pm}
    report["best_delta_layer"] = best_l
    print(f"\n  deep band (L{band[0]}-{band[-1]}): aligned={db_al:.3f} permuted={db_pm:.3f} Δ={db_al-db_pm:.3f}", flush=True)
    print(f"  best Δ layer = {best_l}  (Δ={report['layers'][best_l]['delta']:.3f}, "
          f"aligned={report['layers'][best_l]['aligned']:.3f})", flush=True)

    out_path = args.out or str(Path(__file__).resolve().parent /
                               f"temporal_align_{Path(args.checkpoint).name}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"[done] written {out_path}", flush=True)


if __name__ == "__main__":
    main()
