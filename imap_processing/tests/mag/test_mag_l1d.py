import numpy as np
import pytest
from unittest.mock import patch

from imap_processing.mag.l1d.mag_l1d import mag_l1d
from imap_processing.tests.mag.conftest import mag_l1a_dataset_generator





def test_mag_l1d(norm_dataset, mag_test_l1d_data):
    calibration_dataset = mag_test_l1d_data

    with patch(
        "imap_processing.mag.l2.mag_l2_data.frame_transform",
        side_effect=lambda *args, **kwargs: args[1],
    ):
        l2 = mag_l1d(
            calibration_dataset,
            norm_dataset,
            norm_dataset,
            np.datetime64("2025-10-17"),
        )
    assert "vectors" in l2[0].data_vars
