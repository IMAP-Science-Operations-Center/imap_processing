import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_CMD_ECHO


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_CMD_ECHO.apid[0],
                "filename": "FM45_UltraFM45_Functional_"
                "2024-01-22T0105_20240122T010548.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_process_cmd_echo(decom_test_data, cmd_echo_test_path):
    """Tests process_cmd_echo function."""

    decom_ultra = decom_test_data
    df = pd.read_csv(cmd_echo_test_path, index_col="SequenceCount")

    np.testing.assert_array_equal(
        df.Result, decom_ultra["result_description"].values.flatten()
    )
    np.testing.assert_array_equal(df.Opcode, decom_ultra["opcode"].values.flatten())

    for i, (row, opcode) in enumerate(
        zip(decom_ultra["arguments"].values, decom_ultra["opcode"].values)
    ):
        expected_arg = df.Arguments.values[i].strip()
        expected_len = len(expected_arg.split())

        full_row = np.insert(row, 0, opcode)[:expected_len]
        hex_string = " ".join(f"0x{b:02x}" for b in full_row)

        assert hex_string == expected_arg
