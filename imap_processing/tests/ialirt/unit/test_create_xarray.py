"""Tests xarray based on structure of queried DynamoDB."""

from decimal import Decimal

import numpy as np
import numpy.testing as npt

from imap_processing.cdf.utils import write_cdf
from imap_processing.ialirt.utils.constants import swe_energy
from imap_processing.ialirt.utils.create_xarray import create_xarray_from_records


def test_create_dataset():
    """Tests create_dataset function."""
    records = [
        {
            "instrument": "mag",
            "time_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000001,
            "mag_epoch": 123456789000001,
            "mag_B_GSE": [Decimal("5.0"), Decimal("-3.2"), Decimal("1.1")],
            "mag_B_GSM": [Decimal("4.8"), Decimal("-3.0"), Decimal("1.0")],
            "mag_B_RTN": [Decimal("5.1"), Decimal("-3.3"), Decimal("1.2")],
            "mag_B_magnitude": Decimal("6.0"),
            "mag_phi_B_GSM": Decimal("45.0"),
            "mag_theta_B_GSM": Decimal("30.0"),
            "mag_phi_B_GSE": Decimal("50.0"),
            "mag_theta_B_GSE": Decimal("35.0"),
        },
        {
            "instrument": "codice_hi",
            "time_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000000,
            "codice_hi_epoch": [
                123456789000000,
                123456789000000,
                123456789000000,
                123456789000000,
            ],
            "codice_hi_h": [
                [
                    [[Decimal("1.0") for _ in range(4)] for _ in range(4)]
                    for _ in range(15)
                ]
                for _ in range(4)
            ],
        },
        {
            "instrument": "codice_lo",
            "time_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000000,
            "codice_lo_epoch": 123456789000000,
            "codice_lo_c_over_o_abundance": Decimal("0.5"),
            "codice_lo_mg_over_o_abundance": Decimal("0.3"),
            "codice_lo_fe_over_o_abundance": Decimal("0.2"),
            "codice_lo_c_plus_6_over_c_plus_5": Decimal("0.7"),
            "codice_lo_o_plus_7_over_o_plus_6": Decimal("0.6"),
            "codice_lo_fe_low_over_fe_high": Decimal("0.4"),
        },
        {
            "instrument": "hit",
            "time_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000002,
            "hit_epoch": 123456789000002,
            "hit_e_a_side_low_en": Decimal("0.0"),
            "hit_e_a_side_med_en": Decimal("0.0"),
            "hit_e_a_side_high_en": Decimal("0.0"),
            "hit_e_b_side_low_en": Decimal("0.0"),
            "hit_e_b_side_med_en": Decimal("0.0"),
            "hit_e_b_side_high_en": Decimal("0.0"),
            "hit_h_omni_low_en": Decimal("0.0"),
            "hit_h_omni_med_en": Decimal("0.0"),
            "hit_h_a_side_high_en": Decimal("0.0"),
            "hit_h_b_side_high_en": Decimal("0.0"),
            "hit_he_omni_low_en": Decimal("0.0"),
            "hit_he_omni_high_en": Decimal("0.0"),
        },
        {
            "instrument": "swapi",
            "time_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000002,
            "swapi_epoch": 123456789000002,
            "swapi_pseudo_proton_speed": Decimal("400.0"),
            "swapi_pseudo_proton_density": Decimal("5.0"),
            "swapi_pseudo_proton_temperature": Decimal("100000.0"),
        },
        {
            "instrument": "swe",
            "time_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000002,
            "swe_epoch": 123456789000002,
            "swe_normalized_counts": [Decimal("0.0") for _ in range(8)],
            "swe_counterstreaming_electrons": Decimal("1.0"),
        },
        {
            "instrument": "spacecraft",
            "time_utc": "2025-10-29T18:55:02",
            "ttj2000ns": 123456789000002,
            "sc_position_GSE": [
                Decimal("1373251.6968303905"),
                Decimal("-431299.0150430931"),
                Decimal("73446.43257187483"),
            ],
            "sc_position_GSM": [
                Decimal("1373251.6968303905"),
                Decimal("-400988.5784292875"),
                Decimal("174989.6534196707"),
            ],
            "sc_velocity_GSE": [
                Decimal("0.03919581036966908"),
                Decimal("-0.21796820670587755"),
                Decimal("-0.019698638532273577"),
            ],
            "sc_velocity_GSM": [
                Decimal("0.03919581036966908"),
                Decimal("-1.9156200243319468"),
                Decimal("-3.8606800975317896"),
            ],
        },
    ]

    dataset = create_xarray_from_records(records)

    assert (
        dataset["B_GSM_labels"].values == ["Bx (GSM)", "By (GSM)", "Bz (GSM)"]
    ).all()
    assert (
        dataset["B_RTN_labels"].values
        == ["B radial (RTN)", "B tangential (RTN)", "B normal (RTN)"]
    ).all()
    np.testing.assert_allclose(
        dataset["swe_electron_energy"].values,
        np.array(swe_energy),
        rtol=1e-7,
        atol=1e-6,
    )

    npt.assert_array_equal(
        dataset["swe_normalized_counts"].values[0],
        np.zeros(8, dtype=np.uint32),
    )
    np.testing.assert_allclose(
        dataset["hit_e_a_side_low_en"].values,
        [0.0],
    )
    np.testing.assert_allclose(
        dataset["mag_B_GSE"].sel(mag_epoch=123456789000001).values,
        [5.0, -3.2, 1.1],
    )

    expected_zeros = np.ones((15, 4, 4), dtype=np.float32)
    npt.assert_array_equal(
        dataset["codice_hi_h"].isel(codice_hi_epoch=0).values, expected_zeros
    )

    assert dataset["mag_B_GSE"].dims == ("mag_epoch", "B_GSE_labels")
    assert dataset["swe_normalized_counts"].dims == ("swe_epoch", "swe_electron_energy")
    assert dataset["codice_hi_h"].dims == (
        "codice_hi_epoch",
        "codice_hi_energy_center",
        "codice_hi_spin_sector",
        "codice_hi_elevation",
    )

    # Tests that you can write to a cdf.
    dataset.attrs["Data_version"] = "001"
    dataset.attrs["Start_date"] = "20260114"
    test_data_path = write_cdf(dataset, istp=True, compression=None)

    assert test_data_path.exists()
