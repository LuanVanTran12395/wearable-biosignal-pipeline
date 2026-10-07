"""Step 23 — Quality-assessment reports.

Three CSVs per run, accumulated one row per file:

* **Pipeline QC** — cross-correlation + per-frequency coherence for line-noise
  reduction; RMSE/MAE/SNR/PeakSNR/cross-correlation + per-frequency coherence
  for wavelet thresholding.
* **Data QC** — file length, channel counts, % good channels, rejected channel
  IDs, % variance retained post-wavelet, IC rejection counts, per-segment
  interpolation summary, epoch counts pre/post rejection, retained indices,
  plus per-tag/condition breakdowns where present.
* **Error log** — filename, exception message, traceback for each failed file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


class ReportAccumulator:
    """Collects per-file QC rows and writes the three run-level CSVs."""

    def __init__(self) -> None:
        self.pipeline_rows: List[Dict] = []
        self.data_rows: List[Dict] = []
        self.error_rows: List[Dict] = []

    def add_pipeline(self, filename: str, line_noise: Dict, wavelet: Dict) -> None:
        row: Dict = {"File": filename}
        for k, v in (line_noise or {}).items():
            row[f"lineNoise_{k}"] = v
        for k, v in (wavelet or {}).items():
            row[f"wavelet_{k}"] = v
        self.pipeline_rows.append(row)

    def add_data(self, row: Dict) -> None:
        self.data_rows.append(row)

    def add_error(self, filename: str, message: str, traceback: str) -> None:
        self.error_rows.append({
            "File": filename,
            "Error Message": message,
            "Traceback": traceback,
        })

    def write(self, folder: Path) -> Dict[str, Path]:
        folder.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Path] = {}
        if self.pipeline_rows:
            p = folder / "HAPPE_pipelineQC.csv"
            pd.DataFrame(self.pipeline_rows).to_csv(p, index=False)
            out["pipeline"] = p
        if self.data_rows:
            p = folder / "HAPPE_dataQC.csv"
            pd.DataFrame(self.data_rows).to_csv(p, index=False)
            out["data"] = p
        # Always write the error log (even if empty) so its absence isn't
        # ambiguous — matches HAPPE always emitting the error record.
        p = folder / "HAPPE_errorLog.csv"
        cols = ["File", "Error Message", "Traceback"]
        pd.DataFrame(self.error_rows, columns=cols).to_csv(p, index=False)
        out["errors"] = p
        return out
