"""
UniVi — Visual Modality Unification training toolkit.

Phase 1 public API.
"""

from univi.trainer import (
    train,
    load_dataset,
    build_model,
    apply_lora,
    apply_response_masking,
    make_training_args,
    run_preflight,
    resolve_model_config,
    verify_checkpointing_config,
    detect_markers,
    check_active_labels,
    ZeroActiveLabelsError,
    ActiveLabelCheckCallback,
)
from univi.config import resolve_config, config_hash
from univi.cli import parse_args, MODES
from univi.validation import (
    validate_row_schema,
    validate_dataset,
    select_review_rows,
    generate_review_markdown,
    summarize_validation,
)
from univi.fingerprint import (
    compute_fingerprint,
    fingerprint_hash,
    assert_fingerprints_match,
    FingerprintMismatchError,
)
from univi.evaluation import compute_macro_avg, build_eval_mapping, UniViSFTTrainer
from univi.wandb_utils import WandbManager, validate_wandb_available
from univi.manifest import Manifest, ManifestMismatchError
