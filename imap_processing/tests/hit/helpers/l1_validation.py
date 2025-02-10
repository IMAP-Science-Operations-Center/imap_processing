"""helper functions for HIT L1 unit tests"""

import re

import numpy as np
import pandas as pd
import xarray as xr

# <=== CONSTANTS ===>

# Dictionary of columns to consolidate
#    key = new column name
#    value = prefix of column names in the validation data
RATE_COLUMNS = {
    "coinrates": "COINRATES_",
    "pbufrates": "BUFRATES_",
    "l2fgrates": "L2FGRATES_",
    "l2bgrates": "L2BGRATES_",
    "l3fgrates": "L3FGRATES_",
    "l3bgrates": "L3BGRATES_",
    "penfgrates": "PENFGRATES_",
    "penbgrates": "PENBGRATES_",
    "sectorates": "SECTORATES_",
    "l4fgrates": "L4FGRATES_",
    "l4bgrates": "L4BGRATES_",
    "ialirtrates": "IALIRTRATES_",
    "sngrates_hg": "SNGRATES_HG_",
    "sngrates_lg": "SNGRATES_LG_",
}

# Dictionary of columns to rename
#    key = existing column name
#    value = new column name
RENAME_COLUMNS = {
    "CCSDS_VERSION": "version",
    "CCSDS_TYPE": "type",
    "CCSDS_SEC_HDR_FLAG": "sec_hdr_flg",
    "CCSDS_APPID": "pkt_apid",
    "CCSDS_GRP_FLAG": "seq_flgs",
    "CCSDS_SEQ_CNT": "src_seq_ctr",
    "CCSDS_LENGTH": "pkt_len",
    "SC_TICK": "sc_tick_by_frame",
    "CODE_OK": "hdr_code_ok",
    "HEATER_DUTY_CYCLE": "hdr_heater_duty_cycle",
    "LEAK_CONV": "hdr_leak_conv",
    "DY_TH_STATE": "hdr_dynamic_threshold_state",
    "LIVE_TIME": "livetime_counter",
}

MOD_VALUE_TO_SPECIES_ENERGY_MAP = {
    0: {"species": "H", "energy_idx": 0},
    1: {"species": "H", "energy_idx": 1},
    2: {"species": "H", "energy_idx": 2},
    3: {"species": "He4", "energy_idx": 0},
    4: {"species": "He4", "energy_idx": 1},
    5: {"species": "CNO", "energy_idx": 0},
    6: {"species": "CNO", "energy_idx": 1},
    7: {"species": "NeMgSi", "energy_idx": 0},
    8: {"species": "NeMgSi", "energy_idx": 1},
    9: {"species": "Fe", "energy_idx": 0},
}


# <=== HELPER FUNCTIONS FOR L1 DATA VALIDATION ===>
def prepare_counts_validation_data(validation_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare validation data for comparison with processed data.

    The L1A counts validation data is organized with each value in a
    separate column. This function consolidates related data into
    arrays to match the processed data. It also renames columns to
    match the processed data.

    Parameters
    ----------
    validation_data : pd.DataFrame
        Validation data extracted from a csv file

    Returns
    -------
    pd.DataFrame
        Validation data formatted for comparison with processed data
    """
    validation_data.columns = validation_data.columns.str.strip()
    validation_data.rename(columns=RENAME_COLUMNS, inplace=True)
    validation_data = consolidate_rate_columns(validation_data, RATE_COLUMNS)
    validation_data = process_single_rates(validation_data)
    validation_data = add_species_energy(validation_data)
    validation_data.columns = validation_data.columns.str.lower()
    return validation_data


def prepare_standard_rates_validation_data(
    validation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare validation data for comparison with processed data.

    The L1B standard rates validation data is organized with each
    value in a separate column. This function consolidates related
    data into arrays that match the processed data.

    Parameters
    ----------
    validation_data : pd.DataFrame
        Validation data extracted from a csv file

    Returns
    -------
    pd.DataFrame
        Validation data formatted for comparison with processed data
    """
    validation_data.columns = validation_data.columns.str.strip()
    validation_data = consolidate_rate_columns(
        validation_data, {k: v for k, v in RATE_COLUMNS.items() if k != "sectorates"}
    )
    validation_data = process_single_rates(validation_data)
    validation_data.columns = validation_data.columns.str.lower()
    return validation_data


def consolidate_rate_columns(
    data: pd.DataFrame, rate_columns: dict[str, str]
) -> pd.DataFrame:
    """Consolidate related data into arrays to match processed data.

    The validation data has each value in a separate column. This
    function aggregates related data into arrays to match processed
    data. Each rate column has a corresponding delta plus and delta
    minus column for uncertainty values.

    Parameters
    ----------
    data : pd.DataFrame
        Validation data

    rate_columns : dict[str, str]
        Dictionary of columns to consolidate

    Returns
    -------
    pd.DataFrame
        Validation data with rate columns consolidated into arrays
    """
    for new_col, prefix in rate_columns.items():
        pattern_rates = re.compile(rf"^{prefix}\d+$")
        pattern_delta_plus = re.compile(rf"^{prefix}\d+_DELTA_PLUS$")
        pattern_delta_minus = re.compile(rf"^{prefix}\d+_DELTA_MINUS$")
        data[new_col] = data.filter(regex=pattern_rates.pattern).apply(
            lambda row: row.values, axis=1
        )
        data[f"{new_col}_delta_plus"] = data.filter(
            regex=pattern_delta_plus.pattern
        ).apply(lambda row: row.values, axis=1)
        data[f"{new_col}_delta_minus"] = data.filter(
            regex=pattern_delta_minus.pattern
        ).apply(lambda row: row.values, axis=1)
        if new_col == "sectorates":
            data = consolidate_sectorates(data)
        data.drop(
            columns=data.filter(regex=pattern_rates.pattern).columns, inplace=True
        )
        data.drop(
            columns=data.filter(regex=pattern_delta_plus.pattern).columns, inplace=True
        )
        data.drop(
            columns=data.filter(regex=pattern_delta_minus.pattern).columns, inplace=True
        )
    return data


def consolidate_sectorates(data: pd.DataFrame) -> pd.DataFrame:
    """Consolidate sector rate data into arrays.

    This function distinguishes between sector rate columns with three digits
    and those with four digits in their names.

    SECTORATES_000 SECTORATES_000_0 SECTORATES_000_1 SECTORATES_000_2...SECTORATES_120_9
        0	 	 	 	0
        0	 	 	 	 	             0
        0	 	 	 	 	 	                          0
        :

    Columns with three digits (e.g., SECTORATE_000) contain sectorate
    values for the science frame, with 120 such columns in the validation
    data. These will be organized into an array named "sectorates".

    Columns with four digits (e.g., SECTORATES_000_0) include the sectorate
    values with a mod 10 value appended (e.g., 0). The mod 10 value determines
    the species and energy range the sector rates represent in the science frame.
    There are 10 possible species and energy ranges, but only one has data per
    science frame. The validation data has 10 columns per 120 sector rates,
    totaling 1200 columns per science frame. Each set of 10 columns will have
    only one value, resulting in an array that looks like this:

    [nan, nan, nan, 0, nan, nan, nan, nan, nan, nan...nan, nan, nan, 0, nan...]

    These will be consolidated into a "sectorates_by_mod_val" column in the
    validation data.

    Parameters
    ----------
    data : pd.DataFrame
        Validation data

    Returns
    -------
    pd.DataFrame
        Validation data with sectorate columns consolidated into arrays
    """
    sectorates_three_digits = data.filter(regex=r"^SECTORATES_\d{3}$").columns
    sectorates_delta_plus_three_digits = data.filter(
        regex=r"^SECTORATES_\d{3}_DELTA_PLUS$"
    ).columns
    sectorates_delta_minus_three_digits = data.filter(
        regex=r"^SECTORATES_\d{3}_DELTA_MINUS$"
    ).columns

    data["sectorates"] = data[sectorates_three_digits].apply(
        lambda row: row.values.reshape(8, 15), axis=1
    )
    data["sectorates_delta_plus"] = data[sectorates_delta_plus_three_digits].apply(
        lambda row: row.values.reshape(8, 15), axis=1
    )
    data["sectorates_delta_minus"] = data[sectorates_delta_minus_three_digits].apply(
        lambda row: row.values.reshape(8, 15), axis=1
    )

    sectorates_four_digits = data.filter(regex=r"^SECTORATES_\d{3}_\d{1}$").columns
    data["sectorates_by_mod_val"] = data[sectorates_four_digits].apply(
        lambda row: row.values, axis=1
    )
    data.drop(
        columns=data.filter(regex=r"^SECTORATES_\d{3}_\d{1}.*$").columns, inplace=True
    )
    return data


def process_single_rates(data: pd.DataFrame) -> pd.DataFrame:
    """Combine the high and low gain single rates into 2D arrays

    Parameters
    ----------
    data : pd.DataFrame
        Validation data

    Returns
    -------
    pd.DataFrame
        Validation data with single rates combined into 2D arrays
    """
    data["sngrates"] = data.apply(
        lambda row: np.array([row["sngrates_hg"], row["sngrates_lg"]]), axis=1
    )
    data["sngrates_delta_plus"] = data.apply(
        lambda row: np.array(
            [row["sngrates_hg_delta_plus"], row["sngrates_lg_delta_plus"]]
        ),
        axis=1,
    )
    data["sngrates_delta_minus"] = data.apply(
        lambda row: np.array(
            [row["sngrates_hg_delta_minus"], row["sngrates_lg_delta_minus"]]
        ),
        axis=1,
    )
    data.drop(
        columns=[
            "sngrates_hg",
            "sngrates_lg",
            "sngrates_hg_delta_plus",
            "sngrates_lg_delta_plus",
            "sngrates_hg_delta_minus",
            "sngrates_lg_delta_minus",
        ],
        inplace=True,
    )
    return data


def add_species_energy(data: pd.DataFrame) -> pd.DataFrame:
    """Add species and energy index to the validation data.

    The sector rate data is organized by species and energy index
    in the processed data so this function adds this information
    to each row (i.e. science frame) in the validation data.

    Parameters
    ----------
    data : pd.DataFrame
        Validation data

    Returns
    -------
    pd.DataFrame
        Validation data with species and energy index added
    """
    # Find the mod value for each science frame which equals the
    # first index in the sectorates_by_mod_val array that has a value
    # instead of a nan or empty string.
    # Then use the mod 10 value to determine the species and energy index
    # for each science frame and add this information to the data frame
    data["mod_10"] = data["sectorates_by_mod_val"].apply(
        lambda row: next(
            (i for i, value in enumerate(row) if pd.notna(value) and value != " "), None
        )
    )
    data["species"] = data["mod_10"].apply(
        lambda row: MOD_VALUE_TO_SPECIES_ENERGY_MAP[row]["species"].lower()
        if row is not None
        else None
    )
    data["energy_idx"] = data["mod_10"].apply(
        lambda row: MOD_VALUE_TO_SPECIES_ENERGY_MAP[row]["energy_idx"]
        if row is not None
        else None
    )
    data.drop(columns=["sectorates_by_mod_val", "mod_10"], inplace=True)
    return data


def compare_data(
    expected_data: pd.DataFrame, actual_data: xr.Dataset, skip: list[str]
) -> None:
    """Compare the processed L1A counts data with the validation data.

    Parameters
    ----------
    expected_data : pd.DataFrame
        Validation data extracted from a csv file
        and reformatted for comparison
    actual_data : xr.Dataset
        Processed counts data from l1a processing
    skip : list
        Fields to skip in comparison
    """
    for field in expected_data.columns:
        if field not in [
            "sc_tick_by_frame",
            "species",
            "energy_idx",
        ]:
            assert (
                field in actual_data.data_vars.keys()
            ), f"Field {field} not found in actual data variables"
        if field not in skip:
            for frame in range(expected_data.shape[0]):
                if field == "species":
                    # Compare sector rates data using species and energy index.
                    # which are only present in the validation data. In the actual
                    # data, sector rates are organized by species in 4D arrays.
                    #    i.e. h_counts_sectored has shape
                    #         (epoch, h_energy_index, declination, azimuth).
                    # species and energy index are used to find the correct
                    # array of sector rate data from the actual data for comparison.
                    species = expected_data[field][frame]
                    energy_idx = expected_data["energy_idx"][frame]
                    if "sectorates_delta_plus" in expected_data.columns:
                        np.testing.assert_allclose(
                            actual_data[f"{species}_counts_sectored_delta_plus"][frame][
                                energy_idx
                            ].data,
                            expected_data["sectorates_delta_plus"][frame],
                            rtol=1e-7,  # relative tolerance
                            atol=1e-8,  # absolute tolerance
                            err_msg=f"Mismatch in {species}_counts_sectored_delta_"
                            f"plus at frame {frame}, energy_idx {energy_idx}",
                        )
                    if "sectorates_delta_minus" in expected_data.columns:
                        np.testing.assert_allclose(
                            actual_data[f"{species}_counts_sectored_delta_minus"][
                                frame
                            ][energy_idx].data,
                            expected_data["sectorates_delta_minus"][frame],
                            rtol=1e-7,
                            atol=1e-8,
                            err_msg=f"Mismatch in {species}_counts_sectored_delta_"
                            f"minus at frame {frame}, energy_idx {energy_idx}",
                        )
                    else:
                        np.testing.assert_allclose(
                            actual_data[f"{species}_counts_sectored"][frame][
                                energy_idx
                            ].data,
                            expected_data["sectorates"][frame],
                            rtol=1e-7,
                            atol=1e-8,
                            err_msg=f"Mismatch in {species}_counts_sectored at"
                            f"frame {frame}, energy_idx {energy_idx}",
                        )
                elif field == "sc_tick_by_frame":
                    # Get the sc_tick values for each frame in the actual data
                    # to compare with the validation data
                    sc_tick = actual_data.sc_tick.values
                    sc_tick_by_frame = sc_tick[::20]
                    assert np.array_equal(
                        sc_tick_by_frame[frame], expected_data[field][frame]
                    ), f"Mismatch in {field} at frame {frame}"

                else:
                    np.testing.assert_allclose(
                        actual_data[field][frame].data,
                        expected_data[field][frame],
                        rtol=1e-7,
                        atol=1e-8,
                        err_msg=f"Mismatch in {field} at frame {frame}",
                    )
