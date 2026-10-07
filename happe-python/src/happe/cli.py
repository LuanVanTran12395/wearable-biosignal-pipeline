"""Command-line interface: ``happe run --config config.yaml --input-dir ...``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Params, default_params
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="happe", description="HAPPE EEG pipeline (Python)")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline over a directory of raw files")
    run.add_argument("--config", help="Path to a .yaml/.json Params file")
    run.add_argument("--input-dir", required=True, help="Directory of raw EEG files")
    run.add_argument("--output-dir", required=True, help="Output directory")
    run.add_argument("--erp", action="store_true",
                     help="Use ERP defaults if no --config is given")
    run.add_argument("-v", "--verbose", action="store_true")

    init = sub.add_parser("init-config", help="Write a default config file")
    init.add_argument("--out", required=True, help="Destination .yaml/.json path")
    init.add_argument("--erp", action="store_true", help="ERP defaults")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    if args.command == "init-config":
        params = default_params(erp=args.erp)
        out = Path(args.out)
        if out.suffix.lower() in (".yaml", ".yml"):
            params.to_yaml(str(out))
        else:
            params.to_json(str(out))
        print(f"Wrote default config to {out}")
        return 0

    if args.command == "run":
        params = Params.load(args.config) if args.config else default_params(erp=args.erp)
        report = run_pipeline(params, args.input_dir, args.output_dir)
        n_ok = len(report.data_rows)
        n_err = len(report.error_rows)
        print(f"Done: {n_ok} file(s) processed, {n_err} error(s). "
              f"QC written under '{args.output_dir}/7 - quality_assessment_outputs'.")
        return 1 if n_err and not n_ok else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
