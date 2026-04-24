#!/usr/bin/env python3
"""
main.py — Top-level CLI entry point for AIC-4 Zerone tracker.

Usage examples:
    # Train
    python main.py --config configs/experiments/random_forest.yaml --mode train

    # Evaluate on validation split
    python main.py --config configs/experiments/random_forest.yaml --mode eval

    # Generate Kaggle submission CSV from public-LB sequences
    python main.py --config configs/experiments/random_forest.yaml --mode predict \
                   --output submission.csv

    # Quick check — print the resolved config and registered models, then exit
    python main.py --config configs/experiments/random_forest.yaml --mode info

Registered as a console script in pyproject.toml:
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
    """Train a tracker using the given experiment config."""
    from src.config import load_config
    from src.train import train

    cfg = load_config(cfg_path)
    logging.getLogger(__name__).info(
        "Mode: TRAIN | Config: %s | Model: %s", cfg_path, cfg.model.type
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
    """Evaluate a trained tracker on an annotated split."""
    from src.config import load_config
    from src.inference import evaluate

    cfg = load_config(cfg_path)
    split = args.split or "val"
    logging.getLogger(__name__).info(
        "Mode: EVAL | Config: %s | Split: %s", cfg_path, split
    )
    scores = evaluate(cfg, split=split)

    print("\n── Evaluation Results ───────────────────────────────")
    for key, value in scores.items():
        print(f"  {key:<15}: {value:.4f}")
    print("─────────────────────────────────────────────────────\n")


def _mode_predict(cfg_path: str, args: argparse.Namespace) -> None:
    """Generate a Kaggle submission CSV from public-LB sequences."""
    from src.config import load_config
    from src.inference import predict

    cfg = load_config(cfg_path)
    output_path = args.output or "submission.csv"
    split = args.split or "public_lb"

    logging.getLogger(__name__).info(
        "Mode: PREDICT | Config: %s | Split: %s | Output: %s",
        cfg_path,
        split,
        output_path,
    )
    written = predict(cfg, split=split, output_path=output_path)
    print(f"\n  ✓ Submission written → {written}")
    print(f'  Submit with: make submit FILE={written} MSG="{Path(cfg_path).stem}"\n')


def _mode_info(cfg_path: str, _args: argparse.Namespace) -> None:
    """Print the resolved merged config and the list of registered models."""
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
        description="AIC-4 Zerone — Aerial Single-Object Tracker CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --config configs/experiments/random_forest.yaml --mode train\n"
            "  python main.py --config configs/experiments/csrt_baseline.yaml  --mode eval  --split val\n"
            "  python main.py --config configs/experiments/random_forest.yaml  --mode predict --output sub.csv\n"
            "  python main.py --config configs/experiments/random_forest.yaml  --mode info\n"
        ),
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        metavar="PATH",
        help="Path to the experiment YAML config, e.g. configs/experiments/random_forest.yaml",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["train", "eval", "predict", "info"],
        default="train",
        help="Execution mode (default: train)",
    )
    parser.add_argument(
        "--split",
        metavar="SPLIT",
        default=None,
        help=(
            "Data split to use. "
            "eval mode: 'val' or 'train' (default: val). "
            "predict mode: 'public_lb' (default)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        default=None,
        help="Output path for submission CSV (predict mode only, default: submission.csv)",
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
