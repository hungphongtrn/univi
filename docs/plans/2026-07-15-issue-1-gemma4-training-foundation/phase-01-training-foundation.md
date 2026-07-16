# Phase 1: Training Foundation

## Phase Goal

Build the complete CPU-testable training stack for Gemma 4 E2B visual-modality unification, implementing all 2026-07-15 decisions. After this phase:
- Config normalises YAML + CLI overrides into a resolved config with fingerprint hash
- Dataset loader handles local manifest and Hub pinned-revision paths with subset validation
- Structural/schema validator checks message structure, image-placeholder alignment, and metadata fields
- 32-row stratified processor review produces inspectable markdown and W&B table
- Processor/tokenizer fingerprint is computed (pre-model) and verified (post-model)
- Phase artifact manifests exist with freshness hashes
- Trainer constructor uses `max_length`, `dataset_text_field=""`, `skip_prepare_dataset=true`, `packing=false`, `loss_masking=response_only`
- W&B lifecycle manages init, config logging, artifact upload, and teardown
- CLI framework dispatches `--preflight-only`, `--smoke`, `--train`, `--resume`, `--evaluate` modes
- All tests pass on CPU; GPU tests skip cleanly with `@requires_gpu`

## Phase Exit Criteria

- [ ] `uv run pytest tests/ -v` passes (GPU tests skipped on CPU)
- [ ] `uv run python -m univi.train --help` prints CLI help with all modes
- [ ] `uv run python -m univi.train --config configs/3060_1epoch.yaml --preflight-only` runs preflight (CPU-safe artifact/manifest/config validation) without GPU and exits with status 0 (expects local dataset to exist). Official processor loading is attempted on CPU; if unsupported, the status is recorded as "unavailable_skip" and structural validation/manifest creation proceeds without a pre-model fingerprint. Full official processor/fingerprint verification is gated to GPU smoke (Phase 2).
- [ ] `uv run python -m univi.train --config configs/3060_1epoch.yaml --dry-run` prints resolved config, dataset summary, and fingerprint without side effects
- [ ] `data/validation/<timestamp>/` contains preflight artifacts (validation_summary.json, validation_failures.jsonl, training_data_review.md)
- [ ] All training configs load and normalise correctly through `resolve_config()`: `smoke.yaml`, `3060_full.yaml`, `3060_1epoch.yaml`, `full.yaml`. Validation: `uv run python -c "from univi.config import resolve_config; [resolve_config(f'configs/{n}.yaml') for n in ['smoke','3060_full','3060_1epoch','full']]"` exits 0.
- [ ] `configs/eval.yaml` loads and parses correctly through the eval_lane config loader (not `resolve_config`). Validation: `uv run python -c "from eval_lane import load_eval_config; cfg = load_eval_config('configs/eval.yaml'); assert 'checkpoint_path' in cfg; assert 'eval_datasets' in cfg; assert 'generation' in cfg"` exits 0.
- [ ] The eval.yaml revision-pin update retains its legacy schema and is explicitly excluded from Phase 1 training config normalization. The revision-pin update (adding `model.revision` and `dataset.revision` to eval.yaml) is backward-compatible: `eval_lane.load_eval_config` ignores unknown top-level keys, so the added fields do not break the existing eval harness.
- [ ] No outdated `max_seq_length` references remain in training code path (config files may retain other keys)

## Files to Touch

### New Files

| Path | Purpose |
|------|---------|
| `univi/validation.py` | Structural schema validation, 32-row stratified processor review, sentinel counting |
| `univi/fingerprint.py` | Processor/tokenizer fingerprint computation, pre/post model-load matching |
| `univi/wandb_utils.py` | W&B lifecycle: init, config logging, artifact upload, safe teardown |
| `univi/cli.py` | Argument parser, phase gate logic, mode dispatch |
| `univi/train.py` | CLI entrypoint (`python -m univi.train`): delegates to `univi.cli.main`, supports `--help`, `--preflight-only`, `--dry-run`, `--smoke` as documented. Owned by Task 1 (single CLI/training integration entrypoint module). Tests: import/exit behavior. All Phase 1--3 exit criteria and `train_smoke.py` wrapper point to this owned file. |
| `univi/config.py` | Config schema, defaults, CLI override merging, resolved-config hash |
| `univi/manifest.py` | Phase artifact manifest creation, freshness hash computation and verification |
| `univi/evaluation.py` | Eval helpers: compute_macro_avg, build_eval_mapping, UniViSFTTrainer |

### Modified Files

| Path | Change |
|------|--------|
| `univi/__init__.py` | Export public training API (`train`, `load_dataset`, `build_model`, `resolve_config`, `apply_lora`, `apply_response_masking`, `make_training_args`, `run_preflight`, etc.) |
| `univi/trainer.py` | Full rewrite: new `train()` with `SFTConfig(max_length=2048, ...)`, dataset loader with pinned revisions, `train_on_responses_only`, Unsloth gradient checkpointing propagation, `include_num_input_tokens_seen` |
| `configs/smoke.yaml` | Add `dataset.subsets`, `dataset.train_splits`, `dataset.validation_splits`, `model.revision`, `dataset.revision`, `training.loss_masking`, `training.max_length`, `training.gradient_checkpointing: true`, `training.eval_strategy`, `training.eval_steps`. Remove `max_seq_length`. |
| `configs/3060_full.yaml` | Rewrite: rank 8/alpha 16, `max_length: 2048`, remove inference flags (`gpu_memory_utilization`, `float8_kv_cache`, `unsloth_tiled_mlp`), add dataset splits, model/dataset revision pins, `loss_masking`, `report_to: ["wandb"]` |
| `configs/3060_1epoch.yaml` | Rewrite: same clean-up as 3060_full.yaml, one-epoch num_train_epochs. |
| `configs/full.yaml` | Rewrite: **placeholder/scaled config** — rank 8 is placeholder (not authoritative for A100); expected to be scaled up after Phase 2 smoke. Add revision pins, dataset splits, loss_masking, report_to. Remove max_seq_length. |
| `configs/eval.yaml` | Rewrite: checkpoint paths and revision pins for full evaluation run. |
| `tests/test_training.py` | Rewrite: replace old-contract assertions (`load_config`, `max_seq_length`, rank 32, alpha 64, warmup_steps 20, report_to none) with new API (`resolve_config`, `max_length`, rank 8, alpha 16, warmup_ratio, report_to wandb). Update import-level tests to expect new export names. |
`train_smoke.py` | Rewrite: replace direct trainer imports with thin delegation. Retain `main()` and `if __name__ == "__main__"` block. `main()` calls `from univi.cli import main as cli_main; sys.exit(cli_main(sys.argv[1:]))` — no direct trainer imports, no duplicate dispatch, no argparse, no defaults (passes argv unchanged; see `tests/test_config.py` for `--config` required default). Tests: `uv run python train_smoke.py --config configs/smoke.yaml --help` exits 0; `uv run python train_smoke.py --config configs/smoke.yaml --dry-run` exits 0. Ownership: maintained by Task 1 (CLI framework), used by Task 7 (integration).
`train_full.py` | Rewrite: same as train_smoke.py — thin delegation with no argparse or defaults. Ownership: same as train_smoke.py.
| `tests/test_config.py` | Config schema tests |
| `tests/test_cli.py` | CLI parsing tests |
| `tests/test_manifest.py` | Manifest lifecycle tests |
| `tests/test_dataset.py` | Dataset loader tests |
| `tests/test_validation.py` | Structural validation and 32-row processor review tests |
| `tests/test_fingerprint.py` | Processor/template fingerprint tests |
| `tests/test_trainer.py` | Trainer construction, masking, eval mapping tests |
| `tests/test_wandb.py` | W&B lifecycle tests |
| `tests/test_evaluation.py` | Eval aggregation unit tests (macro_avg, eval mapping, UniViSFTTrainer) |
| `tests/test_integration.py` | CPU-safe integration tests (preflight pipeline, dry-run, config hash stability) |
| `pyproject.toml` | No new dependencies needed (all deps already present). Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `markers = ["requires_gpu: marks GPU-dependent tests"]` if missing. |

### Kept Unchanged

| Path | Rationale |
|------|-----------|
| `data/preprocessing/` | Preprocessing is complete; no training-layer changes needed |
| `data/materialized/` | Dataset is consumed, not produced, by training |
| `eval_lane.py` | Evaluation is Phase 3; no changes now |
| `univi/metrics.py` | Metrics module is for Phase 3 evaluation; no changes now |

## Tasks

---

### Task 1: Config Schema and CLI Framework

**Files:**
- Create: `univi/config.py`
- Create: `univi/cli.py`
- Create: `univi/train.py` (CLI entrypoint: `python -m univi.train`, delegates to `univi.cli.main`, supports --help/--preflight-only/--dry-run/--smoke)
- Create: `univi/manifest.py`
- Create: `tests/conftest.py` (shared fixtures)
- Create: `tests/test_config.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_manifest.py`
- Modify: `univi/__init__.py`

**Step 1.1: Write config schema tests in `tests/test_config.py`**

```python
# --- Config schema tests ---

def test_config_defaults():
    """Default config loads and produces a resolved config hash."""
    from univi.config import resolve_config
    cfg = resolve_config("configs/3060_1epoch.yaml")
    assert cfg["training"]["max_length"] == 2048
    assert "config_hash" in cfg
    assert isinstance(cfg["config_hash"], str)
    assert len(cfg["config_hash"]) == 64  # sha256 hex

def test_config_max_length_not_max_seq_length():
    """Resolved config must use max_length, not max_seq_length."""
    from univi.config import resolve_config
    cfg = resolve_config("configs/3060_1epoch.yaml")
    assert "max_length" in cfg["training"]
    assert "max_seq_length" not in cfg["training"]

def test_config_overrides_merge():
    """CLI overrides merge into resolved config and change hash."""
    from univi.config import resolve_config
    # ... (test with --dataset-source hub, verify cfg override)

def test_config_hub_only_omits_path():
    """Hub-source validation passes when dataset.path is absent and hf_hub_repo_id is set."""
    from univi.config import resolve_config
    cfg = resolve_config("configs/smoke.yaml", overrides={"dataset.path": None, "dataset.hf_hub_repo_id": "hungphongtrn/univi-3M-v0"})
    assert "hf_hub_repo_id" in cfg["dataset"]
```

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (module not found / function not defined)

**Step 1.2: Implement `univi/config.py`**

Implement:
- `resolve_config(yaml_path: str, overrides: dict | None = None) -> dict`
  - Load YAML, apply CLI overrides (deep merge)
  - Validate required keys: `model.name`, `model.revision`, `dataset.subsets`, `training.max_length`
  - Validate conditional keys: `dataset.path` is required when resolved source is `local`; `dataset.hf_hub_repo_id` is required when source is `hub`. When source is `auto`, both are optional (runtime resolves existence).
  - Provide defaults: `loss_masking: "response_only"`, `packing: false`, `report_to: ["wandb"]`
  - Compute `config_hash = sha256(json.dumps(resolved_config, sort_keys=True)).hexdigest()`
  - Return resolved config dict
- `config_hash(cfg: dict) -> str` — deterministic hash of a resolved config
- Schema constants: `REQUIRED_KEYS`, `VALID_DATASET_SOURCES`, `VALID_LOSS_MASKING`, `VALID_MODES`

**Step 1.3: Write failing CLI tests in `tests/test_cli.py`**

```python
def test_cli_parses_modes():
    """CLI parser recognises all explicit modes."""
    from univi.cli import parse_args
    args = parse_args(["--preflight-only"])
    assert args.mode == "preflight_only"

def test_cli_dataset_source_auto():
    """--dataset-source auto resolves correctly."""
    from univi.cli import parse_args
    args = parse_args(["--dataset-source", "hub"])
    assert args.dataset_source == "hub"

def test_cli_default_mode():
    """No mode flag defaults to train."""
    from univi.cli import parse_args
    args = parse_args(["--config", "configs/smoke.yaml"])
    assert args.mode == "train"
```


```python
def test_train_module_imports():
    """univi.train CLI entrypoint module imports delegate cleanly."""
    import univi.train
    assert hasattr(univi.train, "__name__")
```


Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL

**Step 1.4: Implement `univi/cli.py`**

Implement:
- `parse_args(argv: list[str] | None = None) -> argparse.Namespace`
  - `--config CONFIG` (required)
  - `--preflight-only`, `--smoke`, `--resume`, `--evaluate` (mutually exclusive mode group; default: `--train`)
  - `--dataset-source {auto,local,hub}`
  - `--dataset-cache-dir PATH`
  - `--dataset-revision REVISION`
  - `--preflight-workers N`
  - `--dry-run` (validate config + print without executing)
  - `--overrides KEY=VALUE [KEY=VALUE ...]` (YAML key overrides)
- Mode constants: `MODES = ["train", "preflight_only", "smoke", "resume", "evaluate"]`
- `validate_mode(args) -> str` — ensures mode is consistent with args (resume requires --resume, etc.)

Additionally implement:
- `main(argv: list[str] | None = None) -> int` — top-level orchestration:
  1. Parse args via `parse_args(argv)`
  2. If `-h`/`--help` (argparse built-in), print help and return 0
  3. Load YAML from `args.config`, normalize through `resolve_config(yaml_path, overrides=parsed_overrides)`
  4. Dispatch by mode:
     - `--dry-run`: print resolved config (JSON), dataset summary (subset names + row counts when available), fingerprint status, and exit 0. No model loading, no W&B init, no side effects.
     - `--preflight-only`: run preflight (structural validation, review rows, manifests, optional processor fingerprint), print summary, exit 0 on success, 1 on validation failure.
     - `--smoke`: set `training.max_steps` to config's `smoke_steps` (default 10), run training, exit 0 on success, 1 on failure.
     - `--resume`: validate checkpoint directory exists and W&B run ID is recoverable, then launch training from checkpoint. Exit 0 on success, 1 on failure.
     - Default (`--train`): validate preflight and smoke artifacts exist (match manifests), then launch full training. Exit 0 on success, 1 on failure.
  5. Catch unhandled exceptions, print to stderr, return 2
  - Return type is `int` (exit code). Caller (including `univi/train.py`'s `if __name__` block) passes `sys.exit(main(argv))` so help/parse errors produce the correct OS exit code.
  - Mode dispatch never calls `sys.exit()` directly — `main()` returns the code.

Implement `univi/train.py`:
- `#!/usr/bin/env python3` shebang (executable)
- `def main():` — calls `sys.exit(univi.cli.main(sys.argv[1:]))`
- `if __name__ == "__main__": main()`
- No argument parsing, no config loading, no trainer imports — pure delegation to `cli.main`.
- Tests verify `univi.train.__name__` exists (import check) and `univi.train.main()` calls `cli.main` with the correct argv forwarding.

CPU-safe test contract for help and dry-run (add to `tests/test_cli.py`):
```python
def test_cli_help_exits_zero():
    """--help exits 0 via main()."""
    from univi.cli import main
    rc = main(["--help"])
    assert rc == 0

def test_cli_dry_run_known_config(tmp_path):
    """main() with a valid --dry-run config exits 0."""
    from univi.cli import main
    rc = main(["--config", "configs/3060_1epoch.yaml", "--dry-run"])
    assert rc == 0

def test_cli_dry_run_exit_code():
    """Integration: --dry-run exits 0 without GPU (also in test_integration)."""
    from univi.cli import main
    rc = main(["--config", "configs/3060_1epoch.yaml", "--dry-run"])
    assert rc == 0
```

**Step 1.5: Write manifest tests in `tests/test_manifest.py`**

```python
def test_manifest_creates_entry():
    """Creating a manifest entry produces a valid manifest dict."""
    from univi.manifest import Manifest
    m = Manifest(phase="preflight", output_dir="/tmp/test-manifest")
    entry = m.create_entry(
        config_hash="abc123",
        dataset_revision="def456",
        model_revision="4abfca14e6c6bfb5888b80288185b1243fb8d539",
    )
    assert entry["phase"] == "preflight"
    assert "timestamp" in entry
    assert "config_hash" in entry
    assert "model_revision" in entry
    assert entry["model_revision"] == "4abfca14e6c6bfb5888b80288185b1243fb8d539"

def test_manifest_freshness_hash():
    """Manifest hash changes when config hash changes."""
    from univi.manifest import Manifest
    m1 = Manifest(phase="preflight", output_dir="/tmp/test-m1")
    m2 = Manifest(phase="preflight", output_dir="/tmp/test-m2")
    e1 = m1.create_entry(config_hash="aaa", dataset_revision="r1", model_revision="m1")
    e2 = m2.create_entry(config_hash="bbb", dataset_revision="r1", model_revision="m1")
    assert m1.fingerprint(e1) != m2.fingerprint(e2)

def test_manifest_rejects_stale_phase():
    """Loading a preflight manifest when smoke phase expected fails."""
    from univi.manifest import Manifest, ManifestMismatchError
    import tempfile, json
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Manifest(phase="preflight", output_dir=tmpdir)
        entry = m.create_entry(config_hash="abc", dataset_revision="r1", model_revision="m1")
        path = m.save(entry)
        # Now try to load same path expecting a different phase
        m2 = Manifest(phase="smoke", output_dir=tmpdir)
        loaded = m2.load()
        # phase mismatch: preflight vs smoke
        with pytest.raises(ManifestMismatchError, match="phase"):
            m2.verify_freshness(loaded, current_hash="abc")
```

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL

**Step 1.6: Implement `univi/manifest.py`**

Implement:
- `Manifest` class:
  - `__init__(self, phase: str, output_dir: str | Path | None = None)`
    - When `output_dir` is None, saves/loads using current working directory + `manifest.json`
  - `create_entry(config_hash, dataset_revision, model_revision, **extra) -> dict`
    - Returns dict with phase, timestamp, config_hash, dataset_revision, model_revision, and extra fields
  - `save(entry: dict) -> Path` — writes JSON to `output_dir/manifest.json`
  - `load() -> dict` — loads from disk, validates phase and freshness
  - `fingerprint(entry: dict) -> str` — SHA256 of phase-aware subset
  - `verify_freshness(entry: dict, current_hash: str)` — raises `ManifestMismatchError` on mismatch
- Phase constants: `PHASES = ["preflight", "smoke", "train", "resume", "evaluate"]`
- `ManifestMismatchError` exception with phase details

**Step 1.7: Write shared conftest in `tests/conftest.py`**

```python
import pytest
from pathlib import Path

@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a minimal 2-subset dataset at tmp_path with manifest."""
    # ... (create {fineweb-edu: 4 rows, densefusion: 4 rows} + manifest.json)
    return tmp_path

def minimal_config(dataset_path=""):
    """Return a minimal resolved config dict for dataset tests.

    This is a regular Python function (not a pytest fixture) so tests
    can call it directly with optional dataset_path. The returned dict
    includes 'config_hash' for W&B and manifest tests. Owned by Task 1.
    """
    return {
        "model": {"name": "unsloth/gemma-4-E2B-it", "revision": "4abfca14e6c6bfb5888b80288185b1243fb8d539"},
        "dataset": {
            "path": dataset_path or "",
            "subsets": ["fineweb-edu", "densefusion"],
            "train_splits": {"fineweb-edu": ["train"], "densefusion": ["train"]},
            "shuffle_seed": 42,
        },
        "training": {"max_length": 2048},
        "config_hash": "0000000000000000000000000000000000000000000000000000000000000000",
    }
```

**Step 1.8: Run all task 1 tests green**

Run: `uv run pytest tests/test_config.py tests/test_cli.py tests/test_manifest.py -v`
Expected: all PASS
### Task 2: Dataset Loader with Pinned Revisions and Subset Validation

- Modify: `univi/trainer.py` (rewrite `load_dataset`)
- Create: `tests/test_dataset.py`
- (conftest already created in Task 1)

**Step 2.1: Write failing dataset loader tests**

```python
def test_load_dataset_local_detects_manifest():
    """Local loading uses manifest.json when present."""
    from univi.trainer import load_dataset
    # ... (fixture: tmp_path with manifest + child datasets)

def test_load_dataset_local_fallback():
    """Local loading falls back to load_from_disk when no manifest."""
    from univi.trainer import load_dataset
    # ... (fixture: single dataset directory)

def test_load_dataset_hub_pinned_revision():
    """Hub loading uses pinned revision and rejects mismatched ones."""
    from univi.trainer import load_dataset
    cfg = {"dataset": {
        "hf_hub_repo_id": "hungphongtrn/univi-3M-v0",
        "revision": "5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375",
        "subsets": ["fineweb-edu", "densefusion"],
        "train_splits": {"fineweb-edu": ["train"], "densefusion": ["train"]},
    }}
    ds = load_dataset(cfg, source="hub")
    assert len(ds) > 0
    assert set(ds["config_name"]) == {"fineweb-edu", "densefusion"}

def test_load_dataset_rejects_invalid_subset():
    """Unknown subset names raise ValueError."""
    from univi.trainer import load_dataset
    cfg = {"dataset": {
        "path": "data/materialized/univi-3M-v0-split",
        "subsets": ["nonexistent-subset"],
    }}
    with pytest.raises(ValueError, match="Unknown subset"):
        load_dataset(cfg, source="local")

def test_load_dataset_rejects_revision_mismatch_on_resume():
    """Resume with changed dataset revision is rejected."""
    from univi.trainer import load_dataset, RevisionMismatchError
    # ... (mock scenario: manifest has revision X, current config has revision Y)
```

**Step 2.2: Implement dataset loader changes in `univi/trainer.py`**

Rewrite `load_dataset(config: dict, source: str = "auto") -> Dataset`:

```python
VALID_SUBSETS = {"fineweb-edu", "densefusion", "smoltalk", "librispeech"}

def load_dataset(config: dict, source: str = "auto") -> Dataset:
    dcfg = config["dataset"]
    subsets = dcfg.get("subsets", list(VALID_SUBSETS))

    # Validate subset names
    unknown = set(subsets) - VALID_SUBSETS
    if unknown:
        raise ValueError(f"Unknown subset(s): {unknown}. Valid: {VALID_SUBSETS}")

    # Resolve source
    if source == "auto":
        local_path = Path(dcfg.get("path", ""))
        source = "local" if local_path.exists() else "hub"

    if source == "local":
        return _load_local(dcfg, subsets)
    elif source == "hub":
        return _load_hub(dcfg, subsets)
    else:
        raise ValueError(f"Invalid dataset source: {source}")
```

Helpers `_load_local` and `_load_hub` handle manifest resolution, concatenation, shuffle with `seed=dcfg.get("shuffle_seed", 42)`, and revision pinning.

Add `RevisionMismatchError` exception.

**Step 2.3: Consume shared conftest fixtures from Task 1**

Task 1 Step 1.7 already creates `tests/conftest.py` with the `synthetic_dataset` and
`minimal_config` fixtures/helpers. Task 2 consumes these directly:

- `synthetic_dataset` is a `@pytest.fixture` that accepts `tmp_path` and creates
  a minimal 2-subset dataset at the temp path, returning the temp path.
- `minimal_config` is a regular Python function (not a fixture) that returns a
  resolved config dict. Dataset tests call `minimal_config()` or
  `minimal_config(dataset_path=...)` inline — they do NOT require fixture injection.
- Both are owned by Task 1; Task 2 MUST NOT create or modify `tests/conftest.py`.

Task 2's dataset tests import and call these helpers directly in test bodies,
as shown in Step 2.1 above.

**Step 2.4: Run all dataset tests green**
Run: `uv run pytest tests/test_dataset.py -v`
Expected: all PASS. CPU-safe (does not load real Hub dataset — uses `load_dataset` with mocking or fixture data).

---

### Task 3: Structural Validation and 32-Row Processor Review

- Create: `univi/validation.py`
- Create: `tests/test_validation.py`

**Step 3.1: Write failing validation tests**

```python
def test_validator_accepts_valid_row():
    """A well-formed row passes structural validation."""
    from univi.validation import validate_row_schema
    row = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "A cat."}
            ]}
        ],
        "images": [Image.new("RGB", (224, 224))],
        "source_dataset_id": "BAAI/DenseFusion-1M",
        "split": "train",
        "row_id": "0",
        "render_config": "{}",
        "modality_label": "image-text",
        "preprocessing_version": "1.0.0",
    }
    errors = validate_row_schema(row)
    assert len(errors) == 0

def test_validator_rejects_missing_images():
    """Row missing images column fails validation."""
    from univi.validation import validate_row_schema
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}
        ],
        # no "images" column
    }
    errors = validate_row_schema(row)
    assert any("images" in e for e in errors)

def test_validator_rejects_placeholder_count_mismatch():
    """Image placeholder count must match images column length."""
    from univi.validation import validate_row_schema
    row = {..., "images": [img1, img2]}
    # messages has only 1 image placeholder
    errors = validate_row_schema(row)
    assert any("placeholder" in e.lower() for e in errors)

def test_validator_rejects_role_order():
    """Message roles must start with user, then assistant."""
    from univi.validation import validate_row_schema
    row = {..., "messages": [{"role": "assistant", ...}, {"role": "user", ...}]}
    errors = validate_row_schema(row)
    assert any("role" in e.lower() for e in errors)

def test_stratified_selection_eight_per_subset():
    """Stratified selection picks 8 rows from each of 4 subsets = 32 rows."""
    from univi.validation import select_review_rows
    # ... (build dataset with 10 rows per subset)
    rows = select_review_rows(dataset, n_per_subset=8)
    assert len(rows) == 32

def test_validation_summary_contains_expected_counts():
    """Validation summary has raw, valid, failed, and per-subset counts."""
    from univi.validation import summarize_validation
    # ...
    summary = summarize_validation(results)
    assert "raw_rows" in summary
    assert "valid_rows" in summary
    assert "failed_rows" in summary
    assert "per_subset" in summary
    assert "fineweb-edu" in summary["per_subset"]
```

**Step 3.2: Implement `univi/validation.py`**

Implement:

- `validate_row_schema(row: dict) -> list[str]` — returns list of error strings:
  - Checks `messages` is present, length >= 2
  - Checks roles alternate `user`, `assistant`
  - Checks each content entry has valid `type` (`image` or `text`)
  - Counts image placeholders in messages vs `len(row.get("images", []))`
  - Checks required metadata fields exist: `source_dataset_id`, `split`, `row_id`, `render_config`, `modality_label`, `preprocessing_version`
  - Checks no unexpected roles (no `system`)

- `validate_dataset(dataset, max_rows: int = 0) -> ValidationResult`:
  - Iterates rows, collects errors, returns `ValidationResult` with `.valid`, `.failed`, `.errors_per_row`

- `select_review_rows(dataset, n_per_subset: int = 8, seed: int = 42) -> list[dict]`:
  - Groups by `config_name` (subset), selects stratified rows across image-count buckets
  - Returns 32 raw rows with subset metadata

- `generate_review_markdown(rows: list[dict], processor, tokenizer) -> str`:
  - For each row: renders `processor.apply_chat_template(...)`, image references (count), placeholder-vs-image count, target-token count, modality, source, row_id
  - Writes to markdown string suitable for `data/validation/<timestamp>/training_data_review.md`

Run: `uv run pytest tests/test_validation.py -v`
Expected: all PASS

---

### Task 4: Processor/Template Fingerprint

- Create: `univi/fingerprint.py`
- Create: `tests/test_fingerprint.py`

**Step 4.1: Write failing fingerprint tests**

```python
def test_fingerprint_contains_key_fields():
    """Fingerprint dict contains expected keys."""
    from univi.fingerprint import compute_fingerprint
    # ... (mock processor/tokenizer)
    fp = compute_fingerprint(processor, tokenizer)
    assert "tokenizer_vocab_size" in fp
    assert "tokenizer_class" in fp
    assert "special_tokens_map" in fp
    assert "chat_template_hash" in fp
    assert "image_sentinel_ids" in fp
    assert "processor_class" in fp
    assert "image_processor_config" in fp

def test_fingerprint_changes_when_chat_template_changes():
    """Different chat template produces different fingerprint."""
    from univi.fingerprint import compute_fingerprint
    # ... (mock: same processor/tokenizer, different template)
    fp1 = compute_fingerprint(processor1, tokenizer1)
    fp2 = compute_fingerprint(processor2, tokenizer2)
    assert fp1 != fp2

def test_fingerprint_hash_is_stable():
    """Same processor/tokenizer pair produces identical hash."""
    from univi.fingerprint import fingerprint_hash
    fp = {"a": 1, "b": "hello"}
    h1 = fingerprint_hash(fp)
    h2 = fingerprint_hash(fp)
    assert h1 == h2

def test_pre_post_fingerprint_match():
    """Pre- and post-model-load fingerprints must match."""
    from univi.fingerprint import assert_fingerprints_match
    fp1 = {"tokenizer_class": "GemmaTokenizerFast"}
    fp2 = {"tokenizer_class": "GemmaTokenizerFast"}
    assert_fingerprints_match(fp1, fp2)  # no error

def test_pre_post_fingerprint_mismatch_raises():
    """Mismatched fingerprints raise FingerprintMismatchError."""
    from univi.fingerprint import assert_fingerprints_match, FingerprintMismatchError
    fp1 = {"tokenizer_class": "GemmaTokenizerFast"}
    fp2 = {"tokenizer_class": "LlamaTokenizerFast"}
    with pytest.raises(FingerprintMismatchError):
        assert_fingerprints_match(fp1, fp2)
```

**Step 4.2: Implement `univi/fingerprint.py`**

Implement:

- `compute_fingerprint(processor, tokenizer) -> dict`:
  - The `processor` is the Gemma4Processor wrapper returned by
    FastVisionModel.from_pretrained. The `tokenizer` is the wrapper's inner tokenizer
    (processor.tokenizer). The post-model fingerprint MUST be captured after
    get_chat_template(processor.tokenizer, "gemma-4") updates the inner tokenizer's
    template — this ensures the fingerprint reflects the exact tokenizer state used
    during training.
  - `tokenizer_vocab_size`: `len(tokenizer)`
  - `tokenizer_class`: `type(tokenizer).__name__`
  - `special_tokens_map`: json-serializable dict of special tokens
  - `chat_template`: `tokenizer.chat_template` text (or `None`)
  - `chat_template_hash`: `sha256(chat_template or "").hexdigest()`
  - `image_sentinel_ids`: list of ints from processor (or tokenizer) for image tokens
  - `processor_class`: `type(processor).__name__`
  - `processor_config`: config dict (json-serializable)
  - `image_processor_config`: image processor config subset

- `fingerprint_hash(fp: dict) -> str` — SHA256 of sorted JSON

- `assert_fingerprints_match(fp1: dict, fp2: dict)` — raises `FingerprintMismatchError` with diff details

- `fingerprint_diff(fp1: dict, fp2: dict) -> list[str]` — human-readable diff lines

- `FingerprintMismatchError` exception

**Step 4.3: Run fingerprint tests green**

Run: `uv run pytest tests/test_fingerprint.py -v`
Expected: all PASS

---

### Task 5: Trainer Refactor with Latest Decisions

**Files:**
- Modify: `univi/trainer.py` (full `train()` rewrite; add `run_preflight`, `make_training_args`, `apply_response_masking`, `resolve_model_config`, `verify_checkpointing_config`, `detect_markers`; `build_eval_mapping`, `compute_macro_avg`, `UniViSFTTrainer` live in `univi/evaluation.py` (created in Task 5.5 below) and are imported lazily inside `train()` — Task 5.5 must complete before Step 5.4 runs)
- Create: `tests/test_trainer.py`
- Modify: `configs/smoke.yaml`, `configs/3060_full.yaml`, `configs/3060_1epoch.yaml`, `configs/full.yaml`, `configs/eval.yaml`

**Step 5.1: Write failing trainer tests in `tests/test_trainer.py`**

```python
def test_trainer_uses_max_length():
    """SFTConfig uses max_length, not max_seq_length."""
    from univi.trainer import make_training_args
    cfg = {"training": {"max_length": 2048, "output_dir": "/tmp/test"}}
    args = make_training_args(cfg)
    assert args.max_length == 2048

def test_trainer_response_only_masking():
    """train_on_responses_only is called with explicit Gemma 4 markers via tokenizer."""
    from univi.trainer import apply_response_masking
    # ... (mock trainer, mock tokenizer with gemma-4 chat template)
    masked = apply_response_masking(trainer, tokenizer=tokenizer)
    # ... (assert collator has masking callable and labels are masked correctly)

def test_trainer_packing_false():
    """Training packing is explicitly false."""
    from univi.trainer import make_training_args
    cfg = {"training": {"packing": False, "eval_packing": False, "max_length": 2048, "output_dir": "/tmp/test"}}
    args = make_training_args(cfg)
    assert args.packing == False

def test_trainer_skip_prepare_dataset():
    """dataset_kwargs includes skip_prepare_dataset=true."""
    from univi.trainer import make_training_args
    cfg = {"training": {"dataset_text_field": "", "dataset_kwargs": {"skip_prepare_dataset": True}, "max_length": 2048, "output_dir": "/tmp/test"}}
    args = make_training_args(cfg)
    assert args.dataset_text_field == ""
    assert args.dataset_kwargs.get("skip_prepare_dataset") == True

def test_trainer_verify_checkpointing_config():
    """verify_checkpointing_config returns the configured mode."""
    from univi.trainer import verify_checkpointing_config
    cfg = {"model": {"use_gradient_checkpointing": "unsloth"}}
    mode = verify_checkpointing_config(cfg)
    assert mode == "unsloth"

def test_trainer_include_num_input_tokens_seen():
    """include_num_input_tokens_seen is set to non_padding."""
    from univi.trainer import make_training_args
    cfg = {"training": {"include_num_input_tokens_seen": "non_padding", "max_length": 2048, "output_dir": "/tmp/test"}}
    args = make_training_args(cfg)
    assert args.include_num_input_tokens_seen == "non_padding"

def test_trainer_resolve_model_config():
    """resolve_model_config returns model config with pinned revision."""
    from univi.trainer import resolve_model_config
    cfg = {"model": {"name": "unsloth/gemma-4-E2B-it", "revision": "4abfca14e6c6bfb5888b80288185b1243fb8d539"}}
    resolved = resolve_model_config(cfg)
    assert resolved["revision"] == "4abfca14e6c6bfb5888b80288185b1243fb8d539"

def test_detect_markers_gemma():
    """detect_markers returns Gemma 4 expected markers from tokenizer."""
    from univi.trainer import detect_markers
    markers = detect_markers(tokenizer)
    assert markers["user"] == "<|turn>user\n"
    assert markers["model"] == "<|turn>model\n"

def test_zero_active_labels_raises():
    """Batch with zero active response labels raises ZeroActiveLabelsError."""
    from univi.trainer import check_active_labels, ZeroActiveLabelsError
    labels = [-100, -100, -100]
    with pytest.raises(ZeroActiveLabelsError, match="zero active"):
        check_active_labels(labels, batch_idx=0)

def test_zero_active_labels_passes():
    """Batch with active response labels passes."""
    from univi.trainer import check_active_labels
    labels = [-100, -100, 42, 99]
    check_active_labels(labels, batch_idx=0)

def test_gemma_template_renders_expected_markers():
    """Gemma 4 processor.apply_chat_template produces expected user/model delimiters."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("unsloth/gemma-4-E2B-it")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Describe this image."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False)
    assert "<|turn>user\n" in rendered
    assert "<|turn>model\n" in rendered
    assert rendered.count("<|turn>user\n") == 1
    assert rendered.count("<|turn>model\n") == 1


def test_make_training_args_best_model_conditional():
    """metric_for_best_model is only set when validation_subsets are configured."""
    from univi.trainer import make_training_args
    # Without validation_subsets — no best-model params
    cfg_no_eval = {"training": {"max_length": 2048, "output_dir": "/tmp/test"}}
    args_no_eval = make_training_args(cfg_no_eval)
    assert not hasattr(args_no_eval, "metric_for_best_model") or args_no_eval.metric_for_best_model is None
    # With validation_subsets — best-model params present with config defaults
    cfg_eval = {"training": {"max_length": 2048, "output_dir": "/tmp/test", "validation_subsets": ["fineweb-edu"]}}
    args_eval = make_training_args(cfg_eval)
    assert args_eval.metric_for_best_model == "mean_total_loss"
    assert args_eval.greater_is_better == False
    assert args_eval.load_best_model_at_end == True

def test_make_training_args_best_model_from_config():
    """metric_for_best_model reads from config when present."""
    from univi.trainer import make_training_args
    cfg = {"training": {
        "max_length": 2048, "output_dir": "/tmp/test",
        "validation_subsets": ["fineweb-edu"],
        "metric_for_best_model": "eval_loss",
        "greater_is_better": True,
        "load_best_model_at_end": False,
    }}
    args = make_training_args(cfg)
    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better == True
    assert args.load_best_model_at_end == False

def test_make_training_args_best_model_default_precedence():
    """Config keys have explicit defaults when validation_subsets is present but keys are absent."""
    from univi.trainer import make_training_args
    cfg = {"training": {
        "max_length": 2048, "output_dir": "/tmp/test",
        "validation_subsets": ["fineweb-edu"],
        # No metric_for_best_model, greater_is_better, or load_best_model_at_end
    }}
    args = make_training_args(cfg)
    assert args.metric_for_best_model == "mean_total_loss"
    assert args.greater_is_better == False
    assert args.load_best_model_at_end == True
```

Additional trainer tests (model and LoRA construction):

```python

def test_build_model_loads_processor():
    """build_model(cfg) loads processor on CPU when _cpu_validate=True."""
    from univi.trainer import build_model
    cfg = {"model": {"name": "unsloth/gemma-4-E2B-it",
                     "revision": "4abfca14e6c6bfb5888b80288185b1243fb8d539",
                     "max_lora_rank": 8,
                     "use_gradient_checkpointing": "unsloth"},
           "_cpu_validate": True}
    model, processor = build_model(cfg)
    assert model is None  # GPU instantiation skipped on CPU validate
    assert processor is not None
    assert processor.tokenizer is not None  # processor loaded, tokenizer accessible

def test_build_model_config_assertions():
    """build_model raises KeyError when required model config keys are missing."""
    from univi.trainer import build_model
    import pytest
    cfg = {"model": {"name": "unsloth/gemma-4-E2B-it"},  # missing revision, max_lora_rank
           "_cpu_validate": True}
    with pytest.raises(KeyError, match="revision|max_lora_rank|use_gradient_checkpointing"):
        build_model(cfg)

def test_build_model_revision_pinned():
    """build_model validates revision is pinned 40-char hex string."""
    from univi.trainer import build_model
    import pytest
    cfg = {"model": {"name": "unsloth/gemma-4-E2B-it",
                     "revision": "bad-ref",  # not pinned
                     "max_lora_rank": 8,
                     "use_gradient_checkpointing": "unsloth"},
           "_cpu_validate": True}
    with pytest.raises(AssertionError, match="pinned|revision|40"):
        build_model(cfg)

def test_apply_lora_target_modules():
    """apply_lora validates all expected target modules are present."""
    from univi.trainer import apply_lora
    import pytest
    lora_cfg = {"target_modules": ["q_proj", "k_proj", "v_proj"],  # missing o/gate/up/down
                "r": 8, "alpha": 16, "dropout": 0.0, "bias": "none",
                "finetune_audio_layers": False,
                "_cpu_validate": True}
    with pytest.raises(AssertionError, match="target_modules"):
        apply_lora(None, lora_cfg)

def test_apply_lora_cpu_validate():
    """apply_lora with _cpu_validate=True passes on complete valid config."""
    from univi.trainer import apply_lora
    lora_cfg = {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                "r": 8, "alpha": 16, "dropout": 0.0, "bias": "none",
                "finetune_audio_layers": False,
                "finetune_vision_layers": True,
                "finetune_language_layers": True,
                "finetune_attention_layers": True,
                "finetune_mlp_layers": True,
                "_cpu_validate": True}
    result = apply_lora(None, lora_cfg)
    assert result is None  # No GPU model loaded — config-only validation passes

def test_apply_lora_audio_layers_false():
    """apply_lora verifies finetune_audio_layers is explicitly False."""
    from univi.trainer import apply_lora
    import pytest
    lora_cfg = {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                "r": 8, "alpha": 16, "dropout": 0.0, "bias": "none",
                "finetune_audio_layers": True,  # must be False
                "_cpu_validate": True}
    with pytest.raises(AssertionError, match="finetune_audio_layers"):
        apply_lora(None, lora_cfg)

```

**Step 5.2: Implement `univi/trainer.py` rewrite**

The new `train()` function follows this pseudocode. Define these public APIs:

```python
def run_preflight(config: dict, dry_run: bool = False) -> dict:
    """Execute preflight validation without model loading.

    Loads dataset, runs structural validation, selects 32 stratified rows
    for processor review, computes pre-model fingerprint (if processor available),
    initialises W&B (online mode; skipped when dry_run=True),
    generates review markdown and W&B table, uploads validation artifacts,
    and saves the phase manifest.

    When dry_run=True, prints summary without side effects and skips all
    W&B initialisation and artifact upload. Failure during the online
    flow marks the W&B run as failed before teardown.

    Returns dict with 'status', 'validation', 'review_rows', 'fingerprint'.
    """

def resolve_model_config(cfg: dict) -> dict:
    """Extract and validate model configuration dict from resolved config.
    Returns dict with name, revision, load_in_4bit, use_gradient_checkpointing, etc."""

def verify_checkpointing_config(cfg: dict) -> str:
    """Return the effective Unsloth gradient checkpointing mode.
    Validates that use_gradient_checkpointing is one of 'unsloth', True, False.
    Returns the normalized mode string."""

def detect_markers(tokenizer) -> dict:
    """Auto-detect Gemma 4 chat template markers from a tokenizer.

    Receives a tokenizer (must be the inner tokenizer from a FastVisionModel
    processor wrapper, i.e. processor.tokenizer, not the wrapper itself) and
    resolves the user/model delimiter strings from its chat_template field.
    The caller is responsible for applying get_chat_template(processor.tokenizer, "gemma-4")
    and attaching the resolved template to the tokenizer before calling this function.
    Returns dict {'user': str, 'model': str}.
    Raises ValueError if markers cannot be resolved.
    CPU-safe: does not require model weights or GPU.
    """

def check_active_labels(labels: list, batch_idx: int = 0) -> None:
    """Guard: raise ZeroActiveLabelsError if no token has a non-(-100) label."""
    active = [v for v in labels if v != -100 and v is not None]
    if len(active) == 0:
        raise ZeroActiveLabelsError(
            f"Batch {batch_idx} has zero active response labels. "
            "Check marker resolution and response-only masking configuration."
# UniViSFTTrainer and eval helpers are imported lazily inside train()
# to avoid cross-module dependency before univi/evaluation.py is created.


class ZeroActiveLabelsError(Exception):
    """Raised when a training batch has zero active response labels."""

class ActiveLabelCheckCallback(TrainerCallback):
    """Raises ZeroActiveLabelsError if any training batch has zero active labels."""

def make_training_args(cfg: dict) -> SFTConfig:
    """Build SFTConfig from resolved training sub-config.

    Caller (train()) uses a lazy import of UniViSFTTrainer from univi.evaluation
    and constructs it instead of SFTTrainer when validation_subsets are configured.
    UniViSFTTrainer.evaluate() calls super().evaluate(), then inspects the final
    merged metrics dict to compute the unweighted mean of all eval_<subset>_loss
    keys and adds eval_mean_total_loss — the key Trainer resolves from
    metric_for_best_model="mean_total_loss". The subclass is the only mechanism
    that sees all four subset losses after dict-style eval_dataset iteration.

    When validation_subsets is present, reads metric_for_best_model,
    greater_is_better, and load_best_model_at_end from cfg["training"] with
    explicit defaults ("mean_total_loss", False, True). YAML config keys
    take precedence over defaults; these keys are present in the resolved
    config YAML for visibility (see configs/3060_1epoch.yaml).
    """
    tc = cfg["training"]
    args = SFTConfig(
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 4),
        max_length=tc["max_length"],        # NOT max_seq_length
        num_train_epochs=tc.get("num_train_epochs", 1),
        learning_rate=tc.get("learning_rate", 2e-4),
        warmup_ratio=tc.get("warmup_ratio", 0.03),
        warmup_steps=0,                      # ratio is authoritative
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        optim=tc.get("optim", "adamw_8bit"),
        weight_decay=tc.get("weight_decay", 0.001),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        logging_steps=tc.get("logging_steps", 10),
        save_steps=tc.get("save_steps", 1000),
        eval_strategy=tc.get("eval_strategy", "steps"),
        eval_steps=tc.get("eval_steps", 1000),
        output_dir=tc["output_dir"],
        report_to=tc.get("report_to", ["wandb"]),
        remove_unused_columns=False,
        dataloader_num_workers=tc.get("dataloader_num_workers", 0),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        eval_packing=False,
        gradient_checkpointing=tc.get("gradient_checkpointing", True),
        include_num_input_tokens_seen="non_padding",
        save_total_limit=tc.get("save_total_limit", 3),
        seed=tc.get("seed", 3407),
        data_seed=tc.get("data_seed", 42),
    )
    # Conditional best-model selection — only when eval mapping is configured
    eval_subsets = tc.get("validation_subsets", [])
    if eval_subsets:
        args.metric_for_best_model = tc.get("metric_for_best_model", "mean_total_loss")
        args.greater_is_better = tc.get("greater_is_better", False)
        args.load_best_model_at_end = tc.get("load_best_model_at_end", True)
    return args

def train(config: dict, args: argparse.Namespace | None = None) -> SFTTrainer:
    """Execute the training pipeline based on mode from args.

    Dispatches to --preflight-only, --train, --smoke, --resume, or --evaluate
    modes based on args.mode. Each mode documents its own numbered flow below.

    The 'train' arg carries CLI overrides (dataset_source, dry_run, etc.).
    When None, defaults to production train mode with no overrides.
    Returns the constructed/trained SFTTrainer or UniViSFTTrainer instance.
    """
    For --preflight-only:
        1. Load dataset (validate subsets)
        2. Import univi.wandb_utils lazily; init W&B (online mode; skipped
           entirely when args.dry_run is True)
        3. Load processor via FastVisionModel (CPU if possible)
        4. Apply get_chat_template(processor.tokenizer, "gemma-4"); preserve
           resolved template on processor wrapper for collation
        5. Compute pre-model fingerprint; log fingerprint and config_hash to W&B
        6. Run structural validation on all rows (iterate dataset without tokenizing);
           log progress and aggregate counts to W&B
        7. Select 32 stratified rows, run processor review
        8. Generate review markdown and upload W&B table
        9. Upload validation artifacts to W&B (summary, failures, review, manifest)
        10. Compute and save phase manifest
        11. Report validation summary
        12. W&B finish() on success; mark_failed(reason) + finish() on failure
        13. Exit 0
    
    Note: When args.dry_run is True, skip W&B entirely (no init, no logging,
    no artifact upload). In --preflight-only mode, the path attempts
    FastVisionModel processor load on CPU; if the installed Unsloth version
    requires CUDA even for processor extraction, the processor status is
    recorded as "unavailable_skip" and structural validation continues
    without a processor fingerprint. W&B is still init'd for failure
    tracking so the skipped state is observable. Full official processor
    loading and fingerprint matching are gated to GPU smoke (Phase 2).


    For --train (default):
        1. Gate: load preflight manifest via Manifest(phase="preflight").load()
           and smoke manifest via Manifest(phase="smoke").load() — each must
           pass Manifest.verify_freshness(entry, current_hash=cfg["config_hash"])
           and their dataset_revision/model_revision must match the resolved
           config. Abort with a descriptive error naming which artifact is
        2. Load model + processor via FastVisionModel (GPU);
           apply get_chat_template(processor.tokenizer, "gemma-4") to inner
           tokenizer; preserve template on processor wrapper
        3. Apply LoRA (explicit projections, finetune_audio_layers=False)
        4. FastVisionModel.for_training(model)
        5. Compute post-model fingerprint, assert matches pre-model
        6. Construct trainer
           6a. When validation_subsets is non-empty, instantiate UniViSFTTrainer
               (imported lazily inside train() — from univi.evaluation import
               UniViSFTTrainer) instead of SFTTrainer, passing the same
               SFTConfig/args. The overridden evaluate() calls super().evaluate(),
               inspects the final merged mapping with all eval_<subset>_loss keys,
               computes the unweighted mean into eval_mean_total_loss, and returns
               metrics. This MUST happen during trainer construction because the
               Trainer evaluates and saves best checkpoints during the training
               loop, and no callback sees all subset losses after dict-style
               iteration. When validation_subsets is empty, construct SFTTrainer
               directly.
        7. Apply train_on_responses_only via apply_response_masking(trainer, tokenizer)
           — tokenizer is processor.tokenizer (the wrapper's inner tokenizer)
        8. Initialize W&B (or resume)
        9. train()
        10. Save checkpoints, final export, artifacts
        11. Update manifest
        12. W&B teardown
    """
    ...

def build_model(cfg: dict) -> tuple:
    """Load model + processor via Unsloth FastVisionModel with pinned revision.

    Uses FastVisionModel.from_pretrained with the resolved model name and
    revision from cfg["model"]. Configures load_in_4bit, max_lora_rank,
    use_gradient_checkpointing (default "unsloth"), and device_map="auto".

    Returns:
        tuple: (model, processor) — model is the native FastVisionModel instance
        before LoRA application; processor is the loaded AutoProcessor.

    CPU-testable configuration validation:
        When cfg.get("_cpu_validate", False) is True, the function MUST skip GPU
        instantiation and instead return a (None, processor) tuple after asserting
        cfg keys exist (model.name, model.revision, model.max_lora_rank,
        model.use_gradient_checkpointing) and the revision is a pinned 40-char hex
        string. The processor is loaded on CPU and returned for fingerprint tests.

    Ownership: caller MUST verify post-model fingerprint matches pre-model
    fingerprint via assert_fingerprints_match before training.
    Tests: test_build_model_loads_processor, test_build_model_config_assertions,
    test_build_model_revision_pinned.
    """

def apply_lora(model, lora_cfg: dict):
    """Apply LoRA adapters to a loaded model with explicit target projections.

    Targets exactly q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    (from lora_cfg["target_modules"]) with r/alpha/dropout/bias from lora_cfg.
    Always sets finetune_audio_layers=False. Delegates finetune_vision_layers,
    finetune_language_layers, finetune_attention_layers, finetune_mlp_layers from
    lora_cfg (defaults true). After applying adapters, calls
    FastVisionModel.for_training(model).

    Returns:
        The adapted model (same instance, LoRA applied in-place).

    CPU-testable configuration validation:
        When lora_cfg.get("_cpu_validate", False) is True, the function MUST
        assert all expected target_modules are present, r is a positive int,
        alpha is a positive int, dropout in [0.0, 1.0), bias is "none", and
        finetune_audio_layers is explicitly False — without loading any model
        or touching GPU.

    Tests: test_apply_lora_target_modules, test_apply_lora_cpu_validate,
    test_apply_lora_audio_layers_false, test_apply_lora_config_assertions.

    Ownership: called immediately after build_model, before SFTTrainer
    construction and train_on_responses_only.
    """

def apply_response_masking(trainer, tokenizer):
    """Apply train_on_responses_only with Gemma 4 markers.
    
    Uses the correct helper signature: train_on_responses_only takes explicit
    instruction_part/response_part markers and a tokenizer (never processor as
    positional arg). Resolves markers via detect_markers() or uses explicit
    Gemma delimiters. Asserts resolved markers are correct.
    Raises ZeroActiveLabelsError if any batch has zero active labels.
    """
    from unsloth.chat_templates import train_on_responses_only
    # Resolve markers — auto-detect from tokenizer
    markers = detect_markers(tokenizer)
    assert markers["user"] == "<|turn>user\n", f"Unexpected user marker: {markers['user']}"
    assert markers["model"] == "<|turn>model\n", f"Unexpected model marker: {markers['model']}"
    # Call with correct signature: instruction_part, response_part, tokenizer
    train_on_responses_only(
        trainer,
        instruction_part=markers["user"],
        response_part=markers["model"],
        tokenizer=tokenizer,
        force_match=True,
    )
    # Apply zero-active-label guard at batch level
    trainer.add_callback(ActiveLabelCheckCallback(check_active_labels))
```

**Step 5.3: Update configuration files**

Rewrite `configs/3060_1epoch.yaml` as canonical reference:

```yaml
# configs/3060_1epoch.yaml — RTX 3060 12 GB, full data, 1 epoch.
model:
  name: "unsloth/gemma-4-E2B-it"
  revision: "4abfca14e6c6bfb5888b80288185b1243fb8d539"
  load_in_4bit: true
  use_gradient_checkpointing: "unsloth"
  max_lora_rank: 8

lora:
  r: 8
  alpha: 16
  dropout: 0.0
  bias: "none"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  finetune_vision_layers: true
  finetune_language_layers: true
  finetune_attention_layers: true
  finetune_mlp_layers: true
  finetune_audio_layers: false

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  max_length: 2048
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  warmup_steps: 0
  lr_scheduler_type: "cosine"
  weight_decay: 0.001
  max_grad_norm: 1.0
  optim: "adamw_8bit"
  logging_steps: 10
  save_steps: 1000
  eval_strategy: "steps"
  eval_steps: 1000
  output_dir: "data/checkpoints/3060-1epoch-v0"
  report_to: ["wandb"]
  remove_unused_columns: false
  dataloader_num_workers: 0
  packing: false
  eval_packing: false
  loss_masking: "response_only"
  gradient_checkpointing: true
  include_num_input_tokens_seen: "non_padding"
  save_total_limit: 3
  metric_for_best_model: "mean_total_loss"
  greater_is_better: false
  load_best_model_at_end: true
  seed: 3407
  data_seed: 42
  per_device_eval_batch_size: 1
  eval_accumulation_steps: 1
  validation_subsets:
    - "fineweb-edu"
    - "densefusion"
    - "smoltalk"
    - "librispeech"

dataset:
  path: "data/materialized/univi-3M-v0-split"
  revision: "5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375"
  hf_hub_repo_id: "hungphongtrn/univi-3M-v0"
  subsets:
    - "fineweb-edu"
    - "densefusion"
    - "smoltalk"
    - "librispeech"
  train_splits:
    fineweb-edu: ["train"]
    librispeech: ["train"]
    densefusion: ["train"]
    smoltalk: ["train"]
  validation_splits:
    fineweb-edu: ["validation"]
    librispeech: ["validation"]
    densefusion: ["validation"]
    smoltalk: ["validation"]
  shuffle_seed: 42

wandb:
  project: "univi-gemma4"
  entity: null
  run_name_template: "3060-1epoch-v0_{config_hash_short}"

hub:
  model_repo_id: "hungphongtrn/univi-phase0-3060-1epoch-model-v0"
```

Apply the same schema to `smoke.yaml`, `3060_full.yaml`, and `full.yaml`. Differences:
- `smoke.yaml`: `max_steps: 10`, `save_steps: 10`, `output_dir: data/checkpoints/smoke-v0`, no `num_train_epochs`, approximately 10 optimizer steps / checkpoint at step 10
- `3060_full.yaml`: `max_steps: 500`, `save_steps: 100`, `output_dir: data/checkpoints/3060-full-v0`
- `full.yaml`: **Placeholder/scaled config** — rank 8 is NOT authoritative for A100; expected to be scaled up after Phase 2 smoke confirms VRAM headroom. `max_steps: 500`, `save_steps: 100`, `output_dir: data/checkpoints/full-v0`

**`configs/eval.yaml` is NOT processed through the training schema.** It uses the legacy eval_lane schema (top-level: `base_model`, `checkpoint_path`, `eval_datasets`, `output`, `generation`). Its Phase 1 update is limited to revision pins (`model.revision`, `dataset.revision`) added as optional top-level keys that `eval_lane.load_eval_config()` silently ignores. Do NOT pass eval.yaml to `resolve_config()`; validate it separately via `eval_lane.load_eval_config()` (see exit criteria).

**Step 5.4: Run trainer tests green**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: all PASS

Note: Task 5.5 (Evaluation Aggregation Module below) MUST be completed before
this step — trainer.py's lazy import inside train() resolves against
univi/evaluation.py, and the test suite must load trainer.py without
ImportError. Verify Task 5.5.3 passes before running this step.

### Task 5.5: Evaluation Aggregation Module

**Files:**
- Create: `univi/evaluation.py`
- Create: `tests/test_evaluation.py`

**Step 5.5.1: Write failing evaluation tests in `tests/test_evaluation.py`**

```python
def test_compute_macro_avg():
    """compute_macro_avg computes unweighted mean of per-subset NLL losses."""
    from univi.evaluation import compute_macro_avg
    losses = {"fineweb-edu": 1.0, "densefusion": 2.0, "smoltalk": 3.0, "librispeech": 4.0}
    avg = compute_macro_avg(losses)
    assert avg == 2.5

def test_build_eval_mapping():
    """build_eval_mapping creates dict of named eval datasets."""
    from univi.evaluation import build_eval_mapping
    eval_datasets = {"fineweb-edu": ds1, "densefusion": ds2}
    mapping = build_eval_mapping(eval_datasets)
    assert set(mapping.keys()) == {"fineweb-edu", "densefusion"}

def test_uni_vi_sft_trainer_evaluate_adds_mean_total_loss():
    """UniViSFTTrainer.evaluate() adds eval_mean_total_loss to merged metrics."""
    import tests.test_evaluation
    from univi.evaluation import UniViSFTTrainer
    from trl import SFTTrainer
    from unittest.mock import patch, MagicMock
    # Stub the parent evaluate() to return merged per-subset metrics
    trainer = UniViSFTTrainer.__new__(UniViSFTTrainer)
    merged = {"eval_fineweb-edu_loss": 1.5, "eval_densefusion_loss": 2.5, "eval_smoltalk_loss": 3.5, "eval_librispeech_loss": 4.5}
    with patch.object(SFTTrainer, "evaluate", return_value=merged) as mock_parent:
        result = trainer.evaluate()
    assert "eval_mean_total_loss" in result
    assert result["eval_mean_total_loss"] == 3.0

def test_uni_vi_sft_trainer_evaluate_no_subsets():
    """UniViSFTTrainer.evaluate() skips mean when no per-subset keys exist."""
    from univi.evaluation import UniViSFTTrainer
    from trl import SFTTrainer
    from unittest.mock import patch
    trainer = UniViSFTTrainer.__new__(UniViSFTTrainer)
    merged = {"eval_loss": 2.0}  # single overall loss, no per-subset keys
    with patch.object(SFTTrainer, "evaluate", return_value=merged) as mock_parent:
        result = trainer.evaluate()
    assert "eval_mean_total_loss" not in result

def test_uni_vi_sft_trainer_best_model_resolution():
    """UniViSFTTrainer.evaluate() emits eval_mean_total_loss so Trainer's
    metric_for_best_model='mean_total_loss' resolves against the merged dict."""
    from univi.evaluation import UniViSFTTrainer
    from trl import SFTTrainer
    from unittest.mock import patch
    trainer = UniViSFTTrainer.__new__(UniViSFTTrainer)
    merged = {"eval_fineweb-edu_loss": 2.0, "eval_densefusion_loss": 4.0, "eval_smoltalk_loss": 6.0, "eval_librispeech_loss": 8.0}
    with patch.object(SFTTrainer, "evaluate", return_value=merged) as mock_parent:
        result = trainer.evaluate()
    # Simulate Trainer's metric_for_best_model prefix resolution
    metric_to_check = "mean_total_loss"
    if not metric_to_check.startswith("eval_"):
        metric_to_check = f"eval_{metric_to_check}"
    assert metric_to_check in result, f"Trainer looks for '{metric_to_check}'"
    assert result[metric_to_check] == 5.0

def test_uni_vi_sft_trainer_wandb_key_derived():
    """W&B key derivation works on UniViSFTTrainer's output metrics."""
    from univi.evaluation import UniViSFTTrainer
    from trl import SFTTrainer
    from unittest.mock import patch
    trainer = UniViSFTTrainer.__new__(UniViSFTTrainer)
    merged = {"eval_fineweb-edu_loss": 1.5, "eval_densefusion_loss": 2.5, "eval_smoltalk_loss": 3.5, "eval_librispeech_loss": 4.5}
    with patch.object(SFTTrainer, "evaluate", return_value=merged) as mock_parent:
        result = trainer.evaluate()
    # Derive W&B keys with slash namespace
    wandb_metrics = {k.replace("eval_", "eval/", 1): v for k, v in result.items() if k.startswith("eval_")}
    assert "eval/mean_total_loss" in wandb_metrics
    assert wandb_metrics["eval/mean_total_loss"] == 3.0
```


Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL (module not found)

**Step 5.5.2: Implement `univi/evaluation.py`**

Implement:

- `compute_macro_avg(per_subset_losses: dict[str, float]) -> float`:
  - Returns unweighted mean of the values. Returns 0.0 for empty dict.
  - CPU-testable: pure arithmetic.

- `build_eval_mapping(eval_datasets: dict[str, Dataset]) -> dict[str, Dataset]`:
  - Returns the dict as-is; each key produces `eval_<key>_loss` in Trainer metrics.
  - CPU-testable with mock datasets.

- `UniViSFTTrainer(SFTTrainer)`:
  - `evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> dict`:
    - Calls `super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)`
      which recursively evaluates each named subset (when `eval_dataset` is a dict) and returns the final
      merged mapping containing all `eval_<subset>_loss` keys.
    - After the super call returns, scans merged metrics for keys matching `eval_<subset>_loss`.
    - If two or more found, computes the unweighted mean via `compute_macro_avg` and assigns
      `metrics["eval_mean_total_loss"]` — the key the Trainer resolves from
      `metric_for_best_model="mean_total_loss"`.
    - Returns the metrics dict with `eval_mean_total_loss` added. No TrainerControl involvement:
      `evaluate()` returns a dict, not control.
    - When fewer than two per-subset loss keys exist, returns metrics unchanged.
    - CPU-testable: operates on dict values only, no model or GPU required. Tests use
      `unittest.mock.patch.object(SFTTrainer, "evaluate", return_value=merged)` to
      stub the parent path (imported from `trl`).
- W&B key derivation ownership: `WandbManager.log_metrics` is the single owner of
  the metric key namespace transformation. `UniViSFTTrainer.evaluate()` returns
  canonical Trainer metric keys (`eval_<subset>_loss`, `eval_mean_total_loss`) --
  the `eval_` prefix is the Trainer convention for best-model resolution.
  `WandbManager.log_metrics` transforms these immediately before `wandb.log`:
  - `eval_<subset>_loss` -> `eval/<subset>_loss`
  - `eval_mean_total_loss` -> `eval/mean_total_loss`
  This is NOT the subclass's responsibility; `UniViSFTTrainer.evaluate()`
  only sets the `eval_`-prefixed keys that the Trainer uses for best-model selection.
  - `UniViSFTTrainer.evaluate()` sets ONLY `eval_mean_total_loss`, never a bare
    `mean_total_loss`, so there is no ambiguous namespace.
  - Test: `test_wandb_log_metrics_transforms_keys` asserts exact logged keys
    (`eval/fineweb-edu_loss`, `eval/mean_total_loss`) after `WandbManager.log_metrics`.

Ownership: `univi/trainer.py` imports `build_eval_mapping`, `compute_macro_avg`, and `UniViSFTTrainer` from this module via a lazy import inside `train()` — the module-level import was removed to avoid a cross-module dependency before `univi/evaluation.py` is created. The `make_training_args` function sets `metric_for_best_model`, `greater_is_better`, and `load_best_model_at_end` conditionally; the caller (`train()`) is responsible for constructing `UniViSFTTrainer` instead of `SFTTrainer` when `validation_subsets` is non-empty. Task 5.5 MUST complete before Task 5 Step 5.4 runs.

**Step 5.5.3: Run evaluation tests green**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: all PASS. CPU-safe (pure dict operations, no model loading).

---

### Task 6: W&B Lifecycle


- Create: `univi/wandb_utils.py`
- Create: `tests/test_wandb.py`

**Step 6.1: Write failing W&B tests**

```python
def test_wandb_init_with_config():
    """WandbManager.init returns manager with config_hash matching resolved config."""
    from univi.wandb_utils import WandbManager
    cfg = minimal_config()
    mgr = WandbManager.init(cfg, resume=False)
    assert mgr is not None
    assert mgr.config_hash == cfg["config_hash"]
    assert mgr.id is not None

def test_wandb_logs_artifacts():
    """WandbManager.log_validation_artifacts returns per-artifact success dict."""
    from univi.wandb_utils import WandbManager
    cfg = minimal_config()
    mgr = WandbManager.init(cfg)
    result = mgr.log_validation_artifacts(
        summary_path="/tmp/summary.json",
        failures_path="/tmp/failures.jsonl",
        review_path="/tmp/review.md",
        manifest_path="/tmp/manifest.json",
    )
    assert result.get("summary_path")
    assert result.get("failures_path")
    assert result.get("review_path")
    assert result.get("manifest_path")

def test_wandb_resume():
    """Resume with same run_id continues the same W&B run."""
    from univi.wandb_utils import WandbManager
    mgr1 = WandbManager.init(minimal_config(), resume=False)
    run_id = mgr1.id
    mgr1.finish()
    mgr2 = WandbManager.init(minimal_config(), resume=True, run_id=run_id)
    assert mgr2.id == run_id

def test_wandb_mark_failed():
    """mark_failed sets run to failed status without raising."""
    from univi.wandb_utils import WandbManager
    mgr = WandbManager.init(minimal_config())
    mgr.mark_failed("preflight validation failed")
    mgr.finish()

def test_wandb_teardown_on_error():
    """teardown_on_error handles active-run cleanup without raising."""
    from univi.wandb_utils import WandbManager
    mgr = WandbManager.init(minimal_config())
    mgr.teardown_on_error()  # should not raise

def test_wandb_log_metrics_transforms_keys():
    """WandbManager.log_metrics transforms eval_ prefix to eval/ for hierarchical W&B display."""
    from univi.wandb_utils import WandbManager
    import wandb
    from unittest.mock import patch, call
    cfg = minimal_config()
    mgr = WandbManager.init(cfg)
    metrics = {
        "eval_fineweb-edu_loss": 1.5,
        "eval_densefusion_loss": 2.5,
        "eval_mean_total_loss": 2.0,
        "loss": 0.5,
        "grad_norm": 1.2,
    }
    with patch.object(wandb, "log") as mock_log:
        mgr.log_metrics(metrics, step=100)
    expected_calls = [
        call({"eval/fineweb-edu_loss": 1.5, "eval/densefusion_loss": 2.5,
              "eval/mean_total_loss": 2.0, "loss": 0.5, "grad_norm": 1.2}, step=100),
    ]
    mock_log.assert_has_calls(expected_calls)
```

Note: Tests use wandb.init(mode="disabled") or equivalent mock to avoid requiring network credentials.

**Step 6.2: Implement `univi/wandb_utils.py`**

Implement:

- `WandbManager` class:
  - `@classmethod init(cls, cfg: dict, resume: bool = False, run_id: str | None = None) -> WandbManager`
    — Factory: create and initialize (or resume) a W&B run from resolved config.
    Run ID is read from `run_id` on resume; otherwise generated.
    Returns manager instance with `.id` (W&B run ID, str) and `.config_hash` (str).
  - `log_config(resolved_cfg: dict)` — logs flat resolved config to W&B run config.
  - `log_validation_artifacts(summary_path: str, failures_path: str, review_path: str, manifest_path: str) -> dict[str, bool]`
    — Uploads each artifact path; returns dict mapping each path key to upload success.
  - `log_metrics(metrics: dict, step: int | None = None)` -- log training/validation scalars.
    Transforms `eval_<subset>_loss` -> `eval/<subset>_loss` and `eval_mean_total_loss` -> `eval/mean_total_loss`
    immediately before `wandb.log`. All other keys pass through unchanged. This is the single
    authoritative metric namespace transformation point. See W&B key derivation ownership in Task 5.5.2.
  - `mark_failed(reason: str)` — marks the W&B run as failed with reason string.
  - `finish()` — closes the W&B run cleanly.
  - `teardown_on_error()` — safe cleanup on exception; marks run failed if active, then finishes.

- `validate_wandb_available() -> bool` — checks W&B credentials/network without starting a run.

- `WandbConfig` dataclass: project (str), entity (str | None), run_name_template (str), tags (list[str]).

**Step 6.3: Run W&B tests green**

Run: `uv run pytest tests/test_wandb.py -v`
Expected: all PASS

---

### Task 7: CPU Smoke Contract and Integration

- Modify: `univi/__init__.py` (public API — owned by this task)
- Modify: `tests/test_training.py` (rewrite contracts for new config/trainer API)
**Step 7.x: Rewrite `train_smoke.py` and `train_full.py` as thin delegators**

Replace current direct trainer-import content with:

```python
"""
Smoke training entry point for Gemma 4 E2B visual-unification mixture.
Delegates to univi.cli.main for all mode dispatch.

Usage:
    uv run python train_smoke.py --config configs/smoke.yaml
    uv run python train_smoke.py --config configs/smoke.yaml --dry-run
    uv run python train_smoke.py --help
"""
from __future__ import annotations

import sys


def main():
    from univi.cli import main as cli_main
    sys.exit(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
```

Same pattern for `train_full.py` (default `--config` `configs/full.yaml` is NOT set by the wrapper — it relies on `cli.main` logic or the user passing `--config configs/full.yaml`; the wrapper passes argv unchanged).

**Key constraints:**
- `main()` is retained and callable from other modules
- `if __name__ == "__main__"` block is retained
- The wrapper parses ZERO additional arguments — no argparse, no defaults
- The wrapper imports ZERO from `univi.trainer` directly — no `load_config`, `train`, `build_model`, etc.
- The wrapper passes `sys.argv[1:]` verbatim to `cli.main(argv)`
- `cli.main` parses `--config` (required for non-help modes) and dispatches all modes. For `--help`, no `--config` is needed — argparse prints help before required-arg validation.
- Tests: `uv run python train_smoke.py --help` exits 0; `uv run python train_full.py --help` exits 0; `python -c "from train_smoke import main; main()"` with captured argv mock does not raise.

Run: `uv run python train_smoke.py --help` expected: prints help and exits 0.
Run: `uv run python train_full.py --help` expected: prints help and exits 0.
- Create: `tests/test_integration.py`
- (integration tests live in tests/test_integration.py; no separate smoke/ dir needed)

**Step 7.1: Ensure `univi/__init__.py` exports public API**

from univi.trainer import (
    train, load_dataset, build_model, apply_lora,
    apply_response_masking, make_training_args,
    run_preflight, resolve_model_config, verify_checkpointing_config,
    detect_markers, check_active_labels,
    ZeroActiveLabelsError, ActiveLabelCheckCallback,
)
from univi.config import resolve_config, config_hash
from univi.cli import parse_args, MODES
from univi.validation import validate_row_schema, validate_dataset, select_review_rows, generate_review_markdown, summarize_validation
from univi.fingerprint import compute_fingerprint, fingerprint_hash, assert_fingerprints_match, FingerprintMismatchError
from univi.evaluation import compute_macro_avg, build_eval_mapping, UniViSFTTrainer
from univi.wandb_utils import WandbManager, validate_wandb_available
from univi.manifest import Manifest, ManifestMismatchError

**Step 7.2: Write integration tests (CPU-safe)**

```python
def test_full_preflight_pipeline_dry_run(tmp_path, synthetic_dataset):
    """Dry-run preflight produces correct artifacts without model loading."""
    from univi.trainer import run_preflight
    cfg = minimal_config(dataset_path=tmp_path)
    result = run_preflight(cfg, dry_run=True)
    assert result["status"] == "success"
    assert result["validation"]["valid_rows"] >= 0
    assert result["validation"]["failed_rows"] == 0
    # synthetic_dataset fixture has 2 subsets (fineweb-edu, densefusion) x 4 rows each = 8 total
    assert len(result["review_rows"]) == 8
    # Per-subset: select_review_rows yields min(8, available_rows) per subset
    assert sum(1 for r in result["review_rows"] if r["subset"] == "fineweb-edu") == 4
    assert sum(1 for r in result["review_rows"] if r["subset"] == "densefusion") == 4

def test_validation_summary_artifact(tmp_path):
    """Preflight writes summary.json with expected fields."""
    # ... (exercise run_preflight, verify artifact contents)

def test_cli_dry_run_exit_code():
    """--dry-run exits 0 without GPU."""
    from univi.cli import main
    # ... (mock config, run with --dry-run, assert exit 0)

def test_config_hash_stability():
    """Same config produces same hash across runs."""
    from univi.config import resolve_config
    cfg1 = resolve_config("configs/3060_1epoch.yaml")
    cfg2 = resolve_config("configs/3060_1epoch.yaml")
    assert cfg1["config_hash"] == cfg2["config_hash"]

def test_config_hash_changes_with_override():
    """Config hash changes when CLI override changes a value."""
    from univi.config import resolve_config
    cfg1 = resolve_config("configs/3060_1epoch.yaml")
    cfg2 = resolve_config("configs/3060_1epoch.yaml", overrides={"training.max_length": 1024})
    assert cfg1["config_hash"] != cfg2["config_hash"]

def test_all_subset_names_known():
    """All subsets in configs match VALID_SUBSETS."""
    from univi.trainer import VALID_SUBSETS
    for cfg_name in ["smoke", "3060_full", "3060_1epoch", "full"]:
        cfg = resolve_config(f"configs/{cfg_name}.yaml")
        for s in cfg["dataset"]["subsets"]:
            assert s in VALID_SUBSETS, f"{cfg_name}: unknown subset {s}"
```

**Step 7.3: Full test suite pass**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (GPU tests skipped on CPU). Output shows:

```
tests/test_config.py::test_config_defaults PASSED
tests/test_config.py::test_config_overrides_merge PASSED
tests/test_cli.py::test_cli_parses_modes PASSED
tests/test_cli.py::test_cli_default_mode PASSED
...
=== XX passed, YY skipped in Z.ZZs ===
```

**Step 7.4: Dry-run integration test**

Run: `uv run python -m univi.train --config configs/3060_1epoch.yaml --dry-run`
Expected: Prints resolved config hash, dataset summary (subsets and sizes), preflight validation counts, and exits 0. No model loaded, no GPU touched.

**Step 7.5: Preflight integration test (requires local dataset)**

If `data/materialized/univi-3M-v0-split` exists:

```bash
uv run python -m univi.train --config configs/3060_1epoch.yaml --preflight-only
```

Expected: Runs structural validation on all rows, produces `data/validation/<timestamp>/` with:
- `validation_summary.json`
- `validation_failures.jsonl`
- `training_data_review.md` (32 rows, 8 per subset)
- `manifest.json`

Exits 0.

If dataset is not available, this test is skipped (the CPU test suite covers logic without real data).

---

## Open Risks

1. **`FastVisionModel.from_pretrained` on CPU.** The preflight path attempts to load the processor via FastVisionModel on CPU. If the installed Unsloth version requires CUDA even for processor extraction, the processor status is recorded as "unavailable_skip" (no fallback to a different processor API). The preflight-only mode continues with structural validation and manifest creation but does NOT produce a pre-model fingerprint. Official processor loading, fingerprint matching, and collation validation are gated to GPU smoke (Phase 2). This decision is consistent with decisions.md (line 258): "If CPU/off-GPU official load is unsupported, fail rather than silently substitute a different processor API" — the failure is recorded as a documented skip, not a silent substitution. The dry-run mode explicitly avoids model loading, so this risk only affects `--preflight-only`.

2. **`include_num_input_tokens_seen` support.** This feature was added in a specific Transformers/TRL version range and may not be present in the installed TRL 0.24.0. The config must fall back gracefully (omit the flag or accept it as a no-op) and log whether token counting is active.

3. **`max_length` vs `max_seq_length` in TRL SFTConfig.** The installed TRL 0.24.0 may still use `max_seq_length` internally. The config must accept `max_length` as the user-facing key and map it to whatever SFTConfig expects, emitting a deprecation notice if the internal key differs.

4. **W&B credentials on a CPU-only machine (CI).** The W&B lifecycle tests must use `wandb.init(mode="disabled")` or an equivalent mock so they pass without network credentials and without requiring a GPU. The production path enforces online W&B and fails hard if credentials are missing.
## Handoff Notes

- Phase 2 (GPU Smoke Gate) takes the completed `univi.train` CLI, the refactored trainer, and the validated configs, then executes the same code path on a real RTX 3060 to verify collation, masking, loss, W&B, and checkpoint reload.
- The preflight artifacts (validation_summary.json, training_data_review.md) are consumed by Phase 2 as freshness checks.
- The manifest mechanism (Manifest class) enforces that Phase 2 requires Phase 1's preflight manifest to exist and match.
- `--preflight-only` is a CPU-safe artifact/manifest/config validation path: it runs structural validation, selects review rows, creates manifests, and (when processor loading is available) computes pre-model fingerprints. If official `FastVisionModel.from_pretrained` processor load fails on CPU, the status is recorded as "unavailable_skip" without silent substitution. Processor fingerprint verification is gated to GPU smoke (Phase 2). `--dry-run` mode avoids all side effects (no W&B, no model loading, no artifacts).
