import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_PHXTOF_HIGH_TIME


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_PHXTOF_HIGH_TIME.apid[0],
                "filename": "FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_"
                "20240122T093008.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.external_test_data
def test_tof_high_time_decom(decom_test_data, tof_high_time_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    decom_ultra = decom_test_data
    df = pd.read_csv(tof_high_time_test_path, index_col="SequenceCount")

    # Check metadata values
    # The validation csvs provided only includes the last packet's spin value,
    # abortflag, and startdelay
    epoch = len(df["Epoch"].values)
    decom_ultra = decom_ultra.isel(epoch=slice(0, epoch))
    np.testing.assert_array_equal(df.Spin, decom_ultra["spin"][:, -1])
    np.testing.assert_array_equal(df.AbortFlag, decom_ultra["abortflag"][:, -1])
    np.testing.assert_array_equal(df.StartDelay, decom_ultra["startdelay"][:, -1])

    # Validation data from the IT team organizes image data into columns
    # named UltraImage_Plane_Row_Col, where Plane, Row, and Col are 0-indexed.
    # Each row corresponds to the epoch dimension.
    colnames = df.columns.tolist()

    def column_name_sort(name):
        return (
            int(name.split("_")[-3]),
            int(name.split("_")[-2]),
            int(name.split("_")[-1]),
        )

    images = sorted(
        [name for name in colnames if "UltraImage" in name], key=column_name_sort
    )

    epoch = len(df["Epoch"].values)
    planes = max([int(name.split("_")[-3]) for name in images]) + 1
    row = max([int(name.split("_")[-2]) for name in images]) + 1
    col = max([int(name.split("_")[-1]) for name in images]) + 1
    # Reshape the dataframe data into a 4D numpy array
    df_data = df[images].to_numpy().reshape(epoch, planes, row, col)
    # Only check up to the expected number of planes in the decom data
    df_data = df_data[:, : ULTRA_PHXTOF_HIGH_TIME.image_planes, :, :]

    np.testing.assert_array_equal(df_data, decom_ultra["packetdata"])
