import json

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.decom_ultra import decom_image_ena_phxtof_hi_ang_packets


@pytest.fixture()
def decom_ultra(ccsds_path_image_ena_phxtof_hi_ang, xtce_image_ena_phxtof_hi_ang_path):
    """Data for decom_ultra"""
    data_packet_list = decom_image_ena_phxtof_hi_ang_packets(
        ccsds_path_image_ena_phxtof_hi_ang, xtce_image_ena_phxtof_hi_ang_path
    )
    return data_packet_list


def test_image_rate_decom(decom_ultra, image_ena_phxtof_hi_ang_test_path):
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
        # assert (np.array(json.loads(df_data)) == decom_data['packetdata'].data).all()

        # Assuming df_data is a string representation of a list of lists (from your JSON)
        df_data_array = np.array(json.loads(df_data)[0])

        # Ensure the shape and dtype match; you might need to adjust dtype accordingly
        df_data_array = df_data_array.astype(decom_data["packetdata"].data.dtype)

        for i in range(54):
            if (df_data_array[i] == decom_data["packetdata"].data[i]).all() == False:
                print("hi")

        # Compare the arrays
        comparison = df_data_array == decom_data["packetdata"].data

        # Check if all elements are True (i.e., arrays are equal)
        all_equal = comparison.all()

        if not all_equal:
            # Find indices where the arrays do not match
            mismatch_indices = np.where(comparison == False)

            # Print mismatched indices
            print("Mismatched indices:", mismatch_indices)

            # Optional: Print the mismatched values from both arrays for the first few mismatches
            for i in range(
                min(len(mismatch_indices[0]), 5)
            ):  # Adjust the range as needed
                idx = tuple(
                    mismatch_indices[dim][i] for dim in range(len(mismatch_indices))
                )
                print(
                    f"Mismatch at {idx}: df_data_array{idx} = {df_data_array[idx]}, decom_data['packetdata'].data{idx} = {decom_data['packetdata'].data[idx]}"
                )

        print("hi")
