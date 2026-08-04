"""H20 — end-to-end: does the REAL trainer loader return the mixture the configs promise?

`_load_local` reads the manifest, applies the image + token filters and
concatenates every `is_training_split: true` entry. A wrong flag on the extra
`spoken-digits/train` entry would silently DOUBLE the anchor lane; a wrong
max_length would silently drop rows. Both are checked here against the real code.

CPU only.
"""
from __future__ import annotations
import importlib.machinery, importlib.util, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if "unsloth" not in sys.modules:
    try:
        import unsloth  # noqa: F401
    except Exception:
        stub = types.ModuleType("unsloth")
        stub.FastVisionModel = type("FastVisionModel", (), {})
        stub.UnslothVisionDataCollator = type("UnslothVisionDataCollator", (), {})
        stub.is_bfloat16_supported = lambda: False
        stub.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader=None)
        sys.modules["unsloth"] = stub
spec = importlib.util.spec_from_file_location("_tr", ROOT / "univi" / "trainer.py")
tr = importlib.util.module_from_spec(spec); sys.modules["_tr"] = tr; spec.loader.exec_module(tr)

import yaml

for cfg_path in sys.argv[1:]:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    ds = tr.load_dataset(cfg)
    evals = tr._load_eval_datasets(cfg)
    b = cfg["model"]["max_soft_tokens"]
    print(f"\n{cfg_path}  (max_soft_tokens={b}, max_length={cfg['training']['max_length']})")
    print(f"  TRAIN rows returned: {len(ds)}   expected 104014 + 18355 = 122369"
          f"   -> {'OK' if len(ds) == 122369 else 'MISMATCH'}")
    lanes = {}
    for sid in ds.select_columns(["source_dataset_id"])["source_dataset_id"]:
        lanes[sid] = lanes.get(sid, 0) + 1
    print(f"  lanes: {lanes}")
    for name, e in evals.items():
        print(f"  EVAL {name}: {len(e)} rows")
