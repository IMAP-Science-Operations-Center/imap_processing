import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_MACROS_CHECKSUM


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_MACROS_CHECKSUM.apid[0],
                "filename": "FM45_UltraFM45_Functional_"
                "2024-01-22T0105_20240122T010548.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_macrochecksum_decom(
    decom_test_data, ccsds_path_events, xtce_path, macrochecksum_test_path
):
    """Test macroschecksum function."""

    df = pd.read_csv(macrochecksum_test_path, index_col="MET")

    checksum_cols = [col for col in df.columns if col.startswith("Checksum_")]
    df_checksums = df[checksum_cols].copy()

    df_checksums.replace("FILL", 65535, inplace=True)
    df_checksums = df_checksums.astype(np.uint16)

    actual_checksums = decom_test_data["checksum"][0]
    expected_checksums = df_checksums.iloc[0].values

    np.testing.assert_array_equal(actual_checksums, expected_checksums)
