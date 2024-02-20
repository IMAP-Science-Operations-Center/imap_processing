import json

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids


@pytest.fixture()
def decom_ultra(ccsds_path_image_ena_phxtof_hi_ang, xtce_image_ena_phxtof_hi_ang_path):
    """Data for decom_ultra"""
    data_packet_list = decom_ultra_apids(
        ccsds_path_image_ena_phxtof_hi_ang, xtce_image_ena_phxtof_hi_ang_path
    )
    return data_packet_list


def test_image_ena_phxtof_hi_ang_decom(decom_ultra, image_ena_phxtof_hi_ang_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(image_ena_phxtof_hi_ang_test_path, index_col="SequenceCount")

    assert (df["Spin"].values == decom_ultra["spin_data"].values).all()
    assert (df["AbortFlag"].values == decom_ultra["abortflag_data"].values).all()
    assert (df["StartDelay"].values == decom_ultra["startdelay_data"].values).all()
    assert json.loads(df["P00s"].values[0]) == decom_ultra["p00_data"].values[0]

    for count in df.index.get_level_values("SequenceCount").values:
        sid = df[df.index.get_level_values("SequenceCount") == count].SID.values[0]
        epoch = df[df.index.get_level_values("SequenceCount") == count].MET.values[0]
        df_data = df[df.index.get_level_values("SequenceCount") == count].Images.values[
            0
        ]

        decom_data = decom_ultra.sel(measurement={"epoch": epoch, "science_id": sid})
        df_data_array = np.array(json.loads(df_data)[0])

        assert (df_data_array == decom_data["packetdata"].data).all()
