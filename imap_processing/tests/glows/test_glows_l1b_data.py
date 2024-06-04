import pytest

from imap_processing.glows.l1b.glows_l1b_data import AncillaryParameters


def test_glows_l1b_ancillary_file():
    fake_good_input = {
        "version": "0.1",
        "filter_temperature": {
            "min": -30.0,
            "max": 80.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
        "hv_voltage": {
            "min": 0.0,
            "max": 3500.0,
            "n_bits": 12,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
        "spin_period": {"min": 0.0, "max": 20.9712, "n_bits": 16},
        "spin_phase": {"min": 0.0, "max": 360.0, "n_bits": 16},
        "pulse_length": {
            "min": 0.0,
            "max": 255.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
    }

    ancillary = AncillaryParameters(fake_good_input)
    for key in fake_good_input.keys():
        assert getattr(ancillary, key) == fake_good_input[key]

    fake_bad_input = {
        "version": "0.1",
        "filter_temperature": {
            "min": -30.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
    }

    with pytest.raises(KeyError):
        ancillary = AncillaryParameters(fake_bad_input)
