"""Tests xarray based on structure of queried DynamoDB."""

from decimal import Decimal

import numpy as np

from imap_processing.cdf.utils import write_cdf
from imap_processing.ialirt.utils.create_xarray import create_xarray_from_records


def test_create_dataset():
    """Tests create_dataset function."""
    records = [
        {
            "apid": 478,
            "met": 123,
            "utc": "2025-05-21T14:00:00",
            "ttj2000ns": Decimal("111000000000"),
            "hit_e_a_side_low_en": Decimal("1.0"),
            "mag_4s_b_gse": [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")],
        },
        {
            "apid": 478,
            "met": 124,
            "utc": "2025-05-21T15:00:00",
            "ttj2000ns": Decimal("222000000000"),
            "swe_normalized_counts_half_1_esa_0": Decimal("123"),
        },
    ]

    dataset = create_xarray_from_records(records)

    assert (dataset["component"].values == ["x", "y", "z"]).all()

    np.testing.assert_allclose(
        dataset["swe_normalized_counts_half_1_esa_0"].values,
        [4294967295, 123],
    )
    np.testing.assert_allclose(
        dataset["hit_e_a_side_low_en"].values,
        [1.0, 4294967295],
    )
    np.testing.assert_allclose(
        dataset["mag_4s_b_gse"].isel(epoch=0).values,
        [0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        dataset["mag_4s_b_gse"].isel(epoch=1).values,
        [-1.0e31, -1.0e31, -1.0e31],
    )

    assert dataset["mag_4s_b_gse"].dims == ("epoch", "component")

    # Tests that you can write to a cdf.
    dataset.attrs["Data_version"] = "001"
    test_data_path = write_cdf(dataset, istp=True)

    assert test_data_path.exists()
