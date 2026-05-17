#!/usr/bin/env python3
"""
main.py — Top-level CLI entry point for the WhiteBox multimodal cancer XAI pipeline.

Modes
─────
  train    Train a model defined by the experiment YAML config.
  eval     Run the explainability audit on a data split and print a summary.
  predict  Generate XAI artefacts for a split and write a submission CSV.
  info     Print the fully-resolved merged config and registered models, then exit.

Usage examples
──────────────
  # Train the Shimada InceptionV3 WSI model
  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode train

  # Explainability audit on the validation split
  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode eval --split val

  # Generate XAI artefacts for the test split and write submission CSV
  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode predict \\
                 --output submission.csv

  # Same as above but use a specific epoch checkpoint instead of auto-discovery
  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode predict \\
                 --checkpoint logs/runs/checkpoints/shimada2021_replication_e2e_finetune_batch32_weights.pth \\
                 --output submission.csv

  # 'public_lb' is a supported alias for the held-out test split
  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode predict \\
                 --split public_lb --output submission.csv

  # Print the resolved config and all registered model types
  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode info

Output layout (predict / eval)
──────────────────────────────
  XAI artefacts are written to a per-run folder keyed by wandb run_name:
      <explainability_outputs>/<experiment_name>/
          image/raw/image_shap.png
          image/raw/image_lime.png
          image/raw/image_gradcam.png
          image/raw/image_effects.png
          report.json
  The submission CSV is written to --output (default: submission.csv).

Split aliases (predict mode)
─────────────────────────────
  'public_lb', 'lb', 'holdout'  →  mapped to 'test' automatically.

Checkpoint resolution (predict / eval)
────────────────────────────────────────
  Priority 1 — explicit --checkpoint PATH (overrides everything)
  Priority 2 — auto-discovery: <save_dir>/<experiment_name>_weights.pth / _best.pth
  If no checkpoint is found, the warning lists all available .pth / .pkl files
  in save_dir so you can pick one with --checkpoint.

Registered as console scripts in pyproject.toml:
    train-tracker = "main:_train_entry"
    eval-tracker  = "main:_eval_entry"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Logging setup — configure before any src imports so handlers are ready
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


def _mode_train(cfg_path: str, args: argparse.Namespace) -> None:
    """Train a model using the given experiment config.

    Calls src.train.train(cfg) which handles the full training loop,
    checkpointing, W&B logging, and returns final validation scores.
    """
    from src.config import load_config
    from src.train import train

    cfg = load_config(cfg_path)
    logging.getLogger(__name__).info(
        "Mode: TRAIN | Config: %s | Model: %s",
        cfg_path,
        (cfg.get("model") or {}).get("type", "unknown"),
    )
    scores = train(cfg)

    print("\n── Final Scores ─────────────────────────────────────")
    for key, value in scores.items():
        print(f"  {key:<15}: {value:.4f}")
    print("─────────────────────────────────────────────────────")
    print(
        f"\n  → Copy these values into logs/leaderboard.csv\n"
        f"  → Commit with: git commit -m 'docs: update leaderboard — "
        f"{Path(cfg_path).stem} {scores.get('final_score', 0):.3f}'\n"
    )


def _mode_eval(cfg_path: str, args: argparse.Namespace) -> None:
    """Run the explainability pipeline on a named split and print a numeric summary.

    Delegates to src.inference.evaluate which calls run_explainability and
    returns the artefact count / modality breakdown as a float dict.
    Default split: 'val'.
    """
    from src.config import load_config
    from src.inference import evaluate

    cfg = load_config(cfg_path)
    split = args.split or "val"

    # Inject explicit checkpoint into config so _load_model honours it
    # (Priority 1 over auto-discovery by experiment name).
    if getattr(args, "checkpoint", None):
        if not hasattr(cfg, "training"):
            from src.config import DotDict
            cfg.training = DotDict({})
        cfg.training.checkpoint_path = args.checkpoint
        logging.getLogger(__name__).info(
            "Using explicit checkpoint: %s", args.checkpoint
        )

    logging.getLogger(__name__).info(
        "Mode: EVAL | Config: %s | Split: %s", cfg_path, split
    )
    scores = evaluate(cfg, split=split)

    print("\n── Evaluation Results ───────────────────────────────")
    for key, value in scores.items():
        print(f"  {key:<15}: {value:.4f}")
    print("─────────────────────────────────────────────────────\n")

def _mode_auto_audit(cfg_path: str, args: argparse.Namespace) -> None:
    """Intelligently runs both Tabular and Image XAI in two isolated memory passes."""
    import gc
    import torch
    from src.config import load_config
    from src.inference import evaluate

    log = logging.getLogger(__name__)
    split = args.split or "test"

    # Load base configuration
    cfg = load_config(cfg_path)

    # Inject explicit checkpoint if provided
    if getattr(args, "checkpoint", None):
        if not hasattr(cfg, "training"):
            from src.config import DotDict

            cfg.training = DotDict({})
        cfg.training.checkpoint_path = args.checkpoint

    log.info("🚀 STARTING AUTO-AUDIT: Two-Stage Memory-Isolated Execution")

    # ==========================================================
    # STAGE 1: The Genetic Explanations (Fast & Lightweight)
    # ==========================================================
    log.info("========== STAGE 1: Tabular Explanations (SHAP/LIME) ==========")
    cfg.dataset.use_preextracted = True
    cfg.explainability.methods = ["shap", "lime", "pdp", "ice", "ace"]

    try:
        evaluate(cfg, split=split)
    except Exception as e:
        log.error(f"Stage 1 failed: {e}")

    # ==========================================================
    # MEMORY PURGE (The Intelligent Step)
    # ==========================================================
    log.info("🧹 Purging RAM and VRAM before loading Image CNN...")
    gc.collect()  # Force Python to delete orphaned SHAP matrices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Force PyTorch to release VRAM
        torch.cuda.ipc_collect()

    # ==========================================================
    # STAGE 2: The Pathological Explanations (Heavy CNN)
    # ==========================================================
    log.info("========== STAGE 2: Pathological Explanations (Grad-CAM) ==========")
    cfg.dataset.use_preextracted = False
    cfg.explainability.methods = ["gradcam"]

    try:
        evaluate(cfg, split=split)
    except Exception as e:
        log.error(f"Stage 2 failed: {e}")

    log.info("✅ AUTO-AUDIT COMPLETE. All artifacts saved.")

def _mode_predict(cfg_path: str, args: argparse.Namespace) -> None:
    """Run the explainability pipeline and export a submission CSV manifest.

    Split resolution
    ────────────────
    The split defaults to 'test'. Legacy aliases 'public_lb', 'lb', and
    'holdout' are silently mapped to 'test' so old invocations keep working.

    Checkpoint override
    ───────────────────
    If --checkpoint is provided, its path is injected into cfg.training so
    _load_model() in explainability.py picks it up before auto-discovery
    (Priority 1 over the experiment-name-based search).

    Output
    ──────
    XAI artefacts → <explainability_outputs>/<experiment_name>/
    Submission CSV → --output path (default: submission.csv)
    """
    from src.config import load_config
    from src.inference import predict

    cfg = load_config(cfg_path)
    output_path = args.output or "submission.csv"

    # 'public_lb' is a legacy alias for the held-out test split.
    raw_split = args.split or "test"
    _SPLIT_ALIASES = {"public_lb": "test", "lb": "test", "holdout": "test"}
    split = _SPLIT_ALIASES.get(raw_split, raw_split)

    # Inject explicit checkpoint into config so _load_model honours it
    # (Priority 1 over auto-discovery by experiment name).
    if getattr(args, "checkpoint", None):
        if not hasattr(cfg, "training"):
            from src.config import DotDict
            cfg.training = DotDict({})
        cfg.training.checkpoint_path = args.checkpoint

    logging.getLogger(__name__).info(
        "Mode: PREDICT | Config: %s | Split: %s | Output: %s",
        cfg_path,
        split,
        output_path,
    )
    if getattr(args, "checkpoint", None):
        logging.getLogger(__name__).info(
            "Using explicit checkpoint: %s", args.checkpoint
        )
    written = predict(cfg, split=split, output_path=output_path)
    print(f"\n  ✓ Submission written → {written}")
    print(f'  Submit with: make submit FILE={written} MSG="{Path(cfg_path).stem}"\n')


def _mode_info(cfg_path: str, _args: argparse.Namespace) -> None:
    """Print the fully-resolved merged config and all registered model types.

    Useful for verifying that YAML inheritance / overrides are applied
    correctly before kicking off a long training run.
    """
    import json

    from src.config import load_config
    from src.models import list_models

    cfg = load_config(cfg_path)
    print("\n── Resolved Config ──────────────────────────────────")
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    print("\n── Registered Models ────────────────────────────────")
    for name in list_models():
        print(f"  • {name}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="WhiteBox — multimodal cancer explainability CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Train\n"
            "  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode train\n\n"
            "  # Eval on validation split\n"
            "  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode eval --split val\n\n"
            "  # Predict with auto-discovered checkpoint (test split)\n"
            "  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode predict --output submission.csv\n\n"
            "  # Predict with a specific epoch checkpoint\n"
            "  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode predict \\\n"
            "                 --checkpoint logs/runs/checkpoints/shimada2021_replication_e2e_finetune_batch32_weights.pth \\\n"
            "                 --output submission.csv\n\n"
            "  # Inspect resolved config\n"
            "  python main.py --config configs/experiments/shimada_inception_wsi.yaml --mode info\n"
        ),
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        metavar="PATH",
        help="Path to the experiment YAML config, e.g. configs/experiments/shimada_inception_wsi.yaml",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["train", "eval", "predict", "info", "auto_audit"],
        default="train",
        help="Execution mode (train | eval | predict | info | auto_audit; default: train)",
    )
    parser.add_argument(
        "--split",
        metavar="SPLIT",
        default=None,
        help=(
            "Data split to use. "
            "train / eval mode: 'train', 'val' (default: val). "
            "predict mode: 'test' (default) — aliases 'public_lb', 'lb', 'holdout' also accepted."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        default=None,
        help="Destination path for the submission CSV (predict mode only; default: submission.csv)",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        default=None,
        help=(
            "Explicit path to a model checkpoint (.pth / .pkl). "
            "Takes priority over the auto-discovered checkpoint in predict / eval modes. "
            "Example: logs/runs/checkpoints/shimada2021_replication_e2e_finetune_batch32_weights.pth"
        ),
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        help="Logging verbosity: DEBUG | INFO | WARNING (default: INFO)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    cfg_path = args.config
    if not Path(cfg_path).exists():
        log.error("Config file not found: %s", cfg_path)
        return 1

    mode_dispatch = {
        "train": _mode_train,
        "eval": _mode_eval,
        "predict": _mode_predict,
        "info": _mode_info,
        "auto_audit": _mode_auto_audit,  # <-- Added here
    }

    try:
        mode_dispatch[args.mode](cfg_path, args)
    except FileNotFoundError as exc:
        log.error("File not found: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        return 130
    except Exception:
        log.exception("Unexpected error during '%s' mode:", args.mode)
        return 1

    return 0


# ---------------------------------------------------------------------------
# pyproject.toml console-script entry points
# ---------------------------------------------------------------------------


def _train_entry() -> None:
    """Entrypoint for `train-tracker` CLI alias (defaults to --mode train)."""
    sys.exit(main(["--mode", "train"] + sys.argv[1:]))


def _eval_entry() -> None:
    """Entrypoint for `eval-tracker` CLI alias (defaults to --mode eval)."""
    sys.exit(main(["--mode", "eval"] + sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
