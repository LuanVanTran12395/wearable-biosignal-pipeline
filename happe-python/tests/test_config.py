import json

import pytest

from happe.config import Params, default_params
from happe.exceptions import ConfigError


def test_roundtrip_json():
    p = default_params()
    p.lineNoise.freq = 50
    p.chans.labels = ["Cz", "Pz"]
    restored = Params.from_dict(json.loads(p.to_json()))
    assert restored.lineNoise.freq == 50
    assert restored.chans.labels == ["Cz", "Pz"]


def test_roundtrip_yaml(tmp_path):
    p = default_params(erp=True)
    path = tmp_path / "c.yaml"
    p.to_yaml(str(path))
    restored = Params.load(str(path))
    assert restored.paradigm.erp is True
    assert restored.wavelet.threshold_rule == "soft"


def test_validation_rejects_bad_values():
    p = default_params()
    p.reref.method = "nonsense"
    with pytest.raises(ConfigError):
        p.validate()


def test_soft_threshold_allowed_for_resting():
    # DEVIATION (local): original HAPPE restricts soft-threshold to ERP
    # paradigms only; relaxed here to allow an explicit resting-state
    # experiment (see config.py validate() and
    # notebooks/eeg_happe_qc.ipynb). This test documents the relaxed
    # behavior -- soft threshold no longer raises for non-ERP.
    p = default_params(erp=False)
    p.wavelet.threshold_rule = "soft"
    p.validate()  # should not raise
    assert p.wavelet.threshold_rule == "soft"


def test_tuple_fields_survive_roundtrip():
    p = default_params()
    restored = Params.from_dict(p.to_dict())
    assert isinstance(restored.badChans.high_z_pre, tuple)
