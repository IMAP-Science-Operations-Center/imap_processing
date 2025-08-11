"""Tests xarray based on structure of queried DynamoDB."""

from decimal import Decimal

import numpy as np
import numpy.testing as npt

from imap_processing.cdf.utils import write_cdf
from imap_processing.ialirt.utils.create_xarray import create_xarray_from_records


def test_create_dataset():
    """Tests create_dataset function."""
    records = [
        {
            "apid": 478,
            "met": 123456789,
            "met_in_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000000,
            "swe_normalized_counts": [Decimal("0.0") for _ in range(8)],
            "swe_counterstreaming_electrons": Decimal("0.0"),
            "swapi_pseudo_proton_speed": Decimal("0.0"),
            "swapi_pseudo_proton_density": Decimal("0.0"),
            "swapi_pseudo_proton_temperature": Decimal("0.0"),
            "hit_e_a_side_low_en": Decimal("0.0"),
            "hit_e_a_side_med_en": Decimal("0.0"),
            "hit_e_a_side_high_en": Decimal("0.0"),
            "hit_e_b_side_low_en": Decimal("0.0"),
            "hit_e_b_side_med_en": Decimal("0.0"),
            "hit_e_b_side_high_en": Decimal("0.0"),
            "hit_h_omni_med_en": Decimal("0.0"),
            "hit_h_a_side_high_en": Decimal("0.0"),
            "hit_h_b_side_high_en": Decimal("0.0"),
            "hit_he_omni_low_en": Decimal("0.0"),
            "hit_he_omni_high_en": Decimal("0.0"),
            "mag_epoch": Decimal("0.0"),
            "mag_B_GSE": [Decimal("0.0"), Decimal("0.0"), Decimal("0.0")],
            "mag_B_GSM": [Decimal("0.0"), Decimal("0.0"), Decimal("0.0")],
            "mag_B_RTN": [Decimal("0.0"), Decimal("0.0"), Decimal("0.0")],
            "mag_B_magnitude": Decimal("0.0"),
            "mag_phi_B_GSM": Decimal("0.0"),
            "mag_theta_B_GSM": Decimal("0.0"),
            "mag_phi_B_GSE": Decimal("0.0"),
            "mag_theta_B_GSE": Decimal("0.0"),
            "codice_lo_c_over_o_abundance": Decimal("0.0"),
            "codice_lo_mg_over_o_abundance": Decimal("0.0"),
            "codice_lo_fe_over_o_abundance": Decimal("0.0"),
            "codice_lo_c_plus_6_over_c_plus_5_ratio": Decimal("0.0"),
            "codice_lo_o_plus_7_over_o_plus_6_ratio": Decimal("0.0"),
            "codice_lo_fe_low_over_fe_high_ratio": Decimal("0.0"),
            "codice_hi_h": [
                [[Decimal("0.0") for _ in range(4)] for _ in range(4)]
                for _ in range(15)
            ],
        },
        {
            "apid": 478,
            "met": 123456789,
            "met_in_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000001,
            # Only MAG is present
            "mag_epoch": Decimal("0.0"),
            "mag_B_GSE": [Decimal("0.0"), Decimal("0.0"), Decimal("0.0")],
            "mag_B_GSM": [Decimal("0.0"), Decimal("0.0"), Decimal("0.0")],
            "mag_B_RTN": [Decimal("0.0"), Decimal("0.0"), Decimal("0.0")],
            "mag_B_magnitude": Decimal("0.0"),
            "mag_phi_B_GSM": Decimal("0.0"),
            "mag_theta_B_GSM": Decimal("0.0"),
            "mag_phi_B_GSE": Decimal("0.0"),
            "mag_theta_B_GSE": Decimal("0.0"),
        },
        {
            "apid": 478,
            "met": 123456789,
            "met_in_utc": "2025-06-20T08:00:00",
            "ttj2000ns": 123456789000002,
            # Only SWAPI is present
            "swapi_pseudo_proton_speed": Decimal("0.0"),
            "swapi_pseudo_proton_density": Decimal("0.0"),
            "swapi_pseudo_proton_temperature": Decimal("0.0"),
        },
    ]

    dataset = create_xarray_from_records(records)

    assert (dataset["component"].values == ["x", "y", "z"]).all()
    assert (dataset["RTN_component"].values == ["radial", "tangential", "normal"]).all()
    npt.assert_array_equal(dataset["esa_step"].values, np.arange(8))

    npt.assert_array_equal(
        dataset["swe_normalized_counts"].values[0],
        np.zeros(8, dtype=np.uint32),
    )
    npt.assert_array_equal(
        dataset["swe_normalized_counts"].values[1],
        np.full(8, 4294967295, dtype=np.uint32),
    )
    np.testing.assert_allclose(
        dataset["hit_e_a_side_low_en"].values,
        [0, 4294967295, 4294967295],
    )
    np.testing.assert_allclose(
        dataset["mag_B_GSE"].isel(epoch=0).values,
        [0, 0, 0],
    )
    np.testing.assert_allclose(
        dataset["mag_B_GSE"].isel(epoch=1).values,
        [0, 0, 0],
    )

    expected_zeros = np.zeros((15, 4, 4), dtype=np.float32)
    expected_fill = np.full((15, 4, 4), -1e31, dtype=np.float32)

    npt.assert_array_equal(dataset["codice_hi_h"].isel(epoch=0).values, expected_zeros)

    npt.assert_array_equal(dataset["codice_hi_h"].isel(epoch=1).values, expected_fill)

    assert dataset["mag_B_GSE"].dims == ("epoch", "component")
    assert dataset["swe_normalized_counts"].dims == ("epoch", "esa_step")
    assert dataset["codice_hi_h"].dims == (
        "epoch",
        "codice_hi_h_energy_ranges",
        "codice_hi_h_elevation",
        "codice_hi_h_spin_angle",
    )

    # Tests that you can write to a cdf.
    dataset.attrs["Data_version"] = "001"
    test_data_path = write_cdf(dataset, istp=True)

    assert test_data_path.exists()
