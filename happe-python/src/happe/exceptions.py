"""Distinct exception types used across the pipeline.

Each maps to a specific non-fatal failure mode in the MATLAB HAPPE pipeline.
The per-file batch loop (:mod:`happe.pipeline`) catches
:class:`HappeError` (and any other Exception) so that one failing file never
aborts the batch — the error is written to the error-log CSV instead.
"""

from __future__ import annotations


class HappeError(Exception):
    """Base class for all HAPPE pipeline errors."""


class LoadFailError(HappeError):
    """Raised when a raw file cannot be loaded or fails validation.

    Corresponds to MATLAB HAPPE's file-import failure handling (step 1).
    """


class AllICsRejectedError(HappeError):
    """Raised by MuscIL when every independent component is flagged.

    The file becomes unprocessable and is skipped (step 10).
    """


class NoTagsError(HappeError):
    """Raised when none of the configured event tags exist in a file (step 14)."""


class AllTrialsRejectedError(HappeError):
    """Raised when segment rejection would remove every epoch (step 17)."""


class SingleTrialRejectionError(HappeError):
    """Raised when epoch rejection is requested on a single-epoch dataset (step 17)."""


class ConfigError(HappeError):
    """Raised for invalid / inconsistent :class:`~happe.config.Params`."""
