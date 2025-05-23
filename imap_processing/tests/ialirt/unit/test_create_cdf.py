from decimal import Decimal

import numpy as np

from imap_processing.ialirt.utils.create_cdf import create_dataset_from_records


def test_create_dataset_from_records_simple():
    # Example records with two variables and coordinate fields
    records = [
        {   "apid": 478,
            "met": 123,
            "utc": "2025-05-21T14:00:00",
            "ttj2000ns": Decimal("111000000000"),
            "hit_e_a_side_low_en": Decimal("1.0"),
        },
        {
            "apid": 478,
            "met": 124,
            "utc": "2025-05-21T15:00:00",
            "ttj2000ns": Decimal("111000000000"),
            "swe_normalized_counts_quarter_1_esa_0": Decimal("123"),
        },
    ]

    dataset = create_dataset_from_records(records)

    # Coordinate checks
    assert "epoch" in dataset.coords
    assert dataset.dims["epoch"] == 2

    # Variable checks
    assert "swe_normalized_counts_quarter_1_esa_0" in dataset.variables
    assert "hit_e_a_side_low_en" in dataset.variables

    # Value checks
    np.testing.assert_allclose(
        dataset["swe_normalized_counts_quarter_1_esa_0"].values,
        [np.nan, 0.123],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        dataset["hit_e_a_side_low_en"].values,
        [1.0, np.nan],
        equal_nan=True,
    )

    # ttj2000ns handling (optional: treat as variable or coordinate)
    assert "ttj2000ns" in dataset.variables or "ttj2000ns" in dataset.coords
