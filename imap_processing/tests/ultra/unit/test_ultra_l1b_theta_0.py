# TODO: test get_energy_pulse_height
import numpy as np
import pandas as pd
import pytest

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
)
from imap_processing.ultra.l1a.ultra_l1a import create_dataset
from imap_processing.ultra.l1b.ultra_l1b import (
    determine_species_pulse_height,
    get_back_positions,
    get_front_x_position,
    get_front_y_position,
    get_particle_velocity,
    get_path_length,
)

@pytest.fixture()
def decom_ultra_events(ccsds_path_theta_0, xtce_path):
    """Data for decom_ultra_events"""
    data_packet_list = decom_ultra_apids(
        ccsds_path_theta_0, xtce_path, ULTRA_EVENTS.apid[0]
    )
    return data_packet_list


@pytest.fixture()
def decom_ultra_aux(ccsds_path_theta_0, xtce_path):
    """Data for decom_ultra_aux"""
    data_packet_list = decom_ultra_apids(
        ccsds_path_theta_0, xtce_path, ULTRA_AUX.apid[0]
    )
    return data_packet_list


@pytest.fixture()
def indices_start_type_1_or_2(decom_ultra_events, decom_ultra_aux):
    """
    A pytest fixture to extract indices from events_dataset where START_TYPE is 1 or 2
    and COUNT is not 0. Assumes the dataset is structured with an 'epoch' dimension.
    """

    # Create the dataset
    events_dataset = create_dataset(
        {ULTRA_EVENTS.apid[0]: decom_ultra_events,
         ULTRA_AUX.apid[0]: decom_ultra_aux}
    )

    # Remove start_type with fill values
    events_dataset = events_dataset.where(events_dataset["START_TYPE"] !=
                                          GlobalConstants.INT_FILLVAL, drop=True)

    # Check top and bottom
    index_1 = np.where(events_dataset["START_TYPE"] == 1)[0]
    index_2 = np.where(events_dataset["START_TYPE"] == 2)[0]

    return index_1, index_2, events_dataset


@pytest.fixture()
def indices_stop_type_1_or_2(decom_ultra_events, decom_ultra_aux):
    """
    A pytest fixture to extract indices from events_dataset where STOP_TYPE is 1 or 2
    and COUNT is not 0. Assumes the dataset is structured with an 'epoch' dimension.
    """

    # Create the dataset
    events_dataset = create_dataset(
        {ULTRA_EVENTS.apid[0]: decom_ultra_events,
         ULTRA_AUX.apid[0]: decom_ultra_aux}
    )
    # Remove start_type with fill values
    events_dataset = events_dataset.where(events_dataset["START_TYPE"] !=
                                          GlobalConstants.INT_FILLVAL, drop=True)

    # Check top and bottom
    index_1 = np.where(events_dataset["STOP_TYPE"] == 1)[0]
    index_2 = np.where(events_dataset["STOP_TYPE"] == 2)[0]

    return index_1, index_2, events_dataset


def test_xf(
    indices_start_type_1_or_2,
    events_fsw_comparison_theta_0,
):

    indices_1, indices_2, events_dataset = indices_start_type_1_or_2

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices_1]
    selected_rows_2 = df_filt.iloc[indices_2]

    xf_1 = get_front_x_position(events_dataset["START_TYPE"].data[indices_1],
                              events_dataset["START_POS_TDC"].data[indices_1])
    xf_2 = get_front_x_position(events_dataset["START_TYPE"].data[indices_2],
                              events_dataset["START_POS_TDC"].data[indices_2])

    # The value 180 was added to xf_1 since that is the offset from the FSW xft_off
    assert np.allclose(xf_1+180, selected_rows_1.Xf.values.astype('float'), rtol=1e-3)
    # The value 25 was subtracted from xf_2 since that is the offset from the FSW xft_off
    assert np.allclose(xf_2-25, selected_rows_2.Xf.values.astype('float'), rtol=1e-3)


@pytest.fixture()
def tof(indices_stop_type_1_or_2,
        events_fsw_comparison_theta_0):
    indices_1, indices_2, events_dataset = indices_stop_type_1_or_2
    indices = np.concatenate((indices_1,indices_2))
    indices.sort()

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices]

    tof, t2, xb, yb = get_back_positions(
        indices,
        events_dataset,
        selected_rows_1.Xf.values.astype('float')
    )
    return tof, t2, xb, yb


def test_xb_yb(
    indices_stop_type_1_or_2,
    tof,
    events_fsw_comparison_theta_0,
):
    _, _, xb, yb = tof
    indices_1, indices_2, events_dataset = indices_stop_type_1_or_2
    indices = np.concatenate((indices_1,indices_2))

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices]

    np.testing.assert_array_equal(xb[indices],
                                  selected_rows_1["Xb"].astype('float'))
    np.testing.assert_array_equal(yb[indices],
                                  selected_rows_1["Yb"].astype('float'))


def test_yf(
    indices_start_type_1_or_2,
    indices_stop_type_1_or_2,
    events_fsw_comparison_theta_0,
    tof
):
    _, _, events_dataset = indices_start_type_1_or_2

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    d, yf = get_front_y_position(events_dataset,
                                 df_filt.Yb.values.astype('float'))

    assert yf == pytest.approx(df_filt["Yf"].astype('float'), 1e-3)

    xf_test = df_filt["Xf"].astype('float').values
    yf_test = df_filt["Yf"].astype('float').values

    xb_test = df_filt["Xb"].astype('float').values
    yb_test = df_filt["Yb"].astype('float').values

    r = get_path_length((xf_test, yf_test), (xb_test, yb_test), d)
    assert r == pytest.approx(df_filt["r"].astype('float'), rel=1e-3)

    # TODO: test get_energy_pulse_height
    # pulse_height = events_dataset["ENERGY_PH"].data[index]
    # energy = get_energy_pulse_height(pulse_height, stop_type, xb, yb)

    # TODO: needs lookup table to test bin
    tof, t2, xb, yb = tof
    indices_1, indices_2, events_dataset = indices_stop_type_1_or_2
    indices = np.concatenate((indices_1,indices_2))
    indices.sort()

    energy = df_filt["Xf"].iloc[indices].astype('float')
    r = df_filt["Xf"].iloc[indices].astype('float')

    ctof, bin = determine_species_pulse_height(energy, tof[indices], r)
    assert ctof * 100 == pytest.approx(df_filt["cTOF"].iloc[indices].astype('float'), rel=1e-3)
    #
    # energy = float(df["Xf"].iloc[index])
    #
    # # TODO: needs lookup table to test bin
    # ctof, bin = determine_species_pulse_height(energy, tof, r)
    # assert ctof * 100 == pytest.approx(float(df["cTOF"].iloc[index]), rel=1e-3)


def test_positions_3(
    tests_indices,
    events_fsw_comparison_theta_0,
):
    """TODO."""

    indices, events_dataset = tests_indices

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df.replace("FILL", GlobalConstants.INT_FILLVAL, inplace=True)
    selected_rows = df.iloc[indices]


    if events_dataset["STOP_TYPE"].data[index] in [1, 2]:

        energy = float(df["Xf"].iloc[index])



        velocity = get_particle_velocity((xf, yf), (xb, yb), d, tof)

        assert velocity[0] == pytest.approx(
            float(df["vhatX"].iloc[index]), rel=1e-2
        )
        assert velocity[1] == pytest.approx(
            float(df["vhatY"].iloc[index]), rel=1e-2
        )
        assert velocity[2] == pytest.approx(
            float(df["vhatZ"].iloc[index]), rel=1e-2
        )
