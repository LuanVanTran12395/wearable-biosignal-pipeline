"""HAPPE — Harvard Automated Processing Pipeline for EEG (Python port).

Programmatic entry point::

    from happe import run_pipeline, Params
    run_pipeline(params, input_dir="raw/", output_dir="out/")

The step order and per-file error isolation intentionally mirror the MATLAB
HAPPE v4.1 pipeline; see :mod:`happe.pipeline` for the orchestration.
"""

from __future__ import annotations

from .config import Params, default_params
from .exceptions import (
    AllICsRejectedError,
    AllTrialsRejectedError,
    HappeError,
    LoadFailError,
    NoTagsError,
)
from .pipeline import run_pipeline, process_file

__version__ = "4.1.0"

__all__ = [
    "Params",
    "default_params",
    "run_pipeline",
    "process_file",
    "HappeError",
    "LoadFailError",
    "NoTagsError",
    "AllICsRejectedError",
    "AllTrialsRejectedError",
    "__version__",
]
