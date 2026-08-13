"""Command-line entry point for LOCAL_TURBO and DEEP_ONLY modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEEP_TRUST_LEVELS, ON_OFF_CHOICES, DeepConfig, LocalTurboConfig
from .deep.polish import run_deep_only
from .logging_utils import RunLogger
from .pipeline import make_layout, run_local_turbo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-to-epub",
        description="V4 Vietnamese PDF → OCR → EPUB pipeline with optional DeepSeek validation.",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF. In --deep-only mode it is used only to derive the existing output folder name.")
    parser.add_argument("--start", type=int, default=61, help="First PDF page of the existing OCR run (default: 61)")
    parser.add_argument("--end", type=int, default=100, help="Last PDF page of the existing OCR run (default: 100)")
    parser.add_argument("--output", type=Path, default=None, help="Override output directory")
    parser.add_argument("--workers", type=int, default=None, help="Local OCR worker count")
    parser.add_argument("--batch-sides", type=int, default=2, help="Whole-side images per Tesseract process (default: 2)")
    parser.add_argument("--tesseract", type=str, default=None, help="Explicit tesseract executable")

    parser.add_argument("--deep-only", action="store_true", help="Skip PDF/Tesseract completely and polish existing LOCAL_TURBO output")
    parser.add_argument("--deep-start", type=int, default=None, help="Optional first PDF page to send to Deep while reusing the full OCR run")
    parser.add_argument("--deep-end", type=int, default=None, help="Optional last PDF page to send to Deep while reusing the full OCR run")
    parser.add_argument("--model", default=DeepConfig().model, help="OpenCode model for --deep-only")
    parser.add_argument("--ai-batch-size", type=int, default=6, help="Deep-only items per model call (default: 6)")
    parser.add_argument("--ai-workers", type=int, default=4, help="Parallel Deep-only calls (default: 4)")
    parser.add_argument(
        "--deep-trust",
        choices=DEEP_TRUST_LEVELS,
        default=DeepConfig().deep_trust,
        help="Deep correction authority: strict, balanced, or high (default: high)",
    )
    parser.add_argument(
        "--ocr-evidence-gate",
        choices=ON_OFF_CHOICES,
        default="on",
        help="Use OCR candidates to veto Deep corrections: on/off (default: on)",
    )
    parser.add_argument(
        "--patch-projection",
        choices=ON_OFF_CHOICES,
        default="on",
        help="Project accepted edits back onto original OCR line breaks: on/off (default: on)",
    )
    parser.add_argument(
        "--min-apply-confidence",
        type=float,
        default=0.97,
        help="Minimum Deep confidence when OCR evidence gate is disabled; evidence baseline otherwise",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = make_layout(args.pdf, args.start, args.end, args.output)
    layout.root.mkdir(parents=True, exist_ok=True)

    if args.deep_only:
        if (args.deep_start is None) != (args.deep_end is None):
            raise SystemExit("--deep-start and --deep-end must be supplied together")
        if args.deep_start is not None:
            if args.deep_start > args.deep_end:
                raise SystemExit("--deep-start must be <= --deep-end")
            if args.deep_start < args.start or args.deep_end > args.end:
                raise SystemExit("Deep test range must stay inside the existing OCR --start/--end range")

        config = DeepConfig(
            model=args.model,
            batch_size=args.ai_batch_size,
            workers=args.ai_workers,
            min_apply_confidence=args.min_apply_confidence,
            deep_trust=args.deep_trust,
            ocr_evidence_gate=args.ocr_evidence_gate == "on",
            patch_projection=args.patch_projection == "on",
        )
        if args.deep_start is None:
            log_name = "run_deep_only.log"
        else:
            log_name = f"run_deep_only_{args.deep_start:03d}_{args.deep_end:03d}.log"
        with RunLogger(layout.root / log_name) as logger:
            logger.log("=== DEEP-ONLY MICRO-BATCH POLISH ===")
            logger.log(f"Source: {layout.root}")
            logger.log("OCR/Tesseract: SKIPPED (reuse existing FIX3 output)")
            logger.log(f"Model: {config.model}")
            logger.log(f"Deep trust: {config.deep_trust}")
            logger.log(f"OCR evidence gate: {'on' if config.ocr_evidence_gate else 'off'}")
            logger.log(f"Patch projection: {'on' if config.patch_projection else 'off'}")
            run_deep_only(
                layout,
                config,
                logger.log,
                page_start=args.deep_start,
                page_end=args.deep_end,
            )
            logger.log("=== DONE DEEP-ONLY ===")
        return 0

    local = LocalTurboConfig(
        start_page=args.start,
        end_page=args.end,
        workers=args.workers or LocalTurboConfig().workers,
        batch_sides=args.batch_sides,
    )
    with RunLogger(layout.root / "run_v4_local_turbo.log") as logger:
        run_local_turbo(args.pdf, layout, local, logger.log, tesseract_cmd=args.tesseract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
