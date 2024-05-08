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
def events_dataset(
    decom_ultra_events,
    decom_ultra_aux,
):
    """TODO."""

    events_dataset = create_dataset(
        {ULTRA_EVENTS.apid[0]: decom_ultra_events, ULTRA_AUX.apid[0]: decom_ultra_aux}
    )

    return events_dataset


@pytest.fixture()
def tof(
    decom_ultra_events,
    decom_ultra_aux,
):
    """TODO."""

    events_dataset = create_dataset(
        {ULTRA_EVENTS.apid[0]: decom_ultra_events, ULTRA_AUX.apid[0]: decom_ultra_aux}
    )

    return events_dataset


def test_positions(
    events_dataset,
    events_fsw_comparison_theta_0,
):
    """TODO."""

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df.replace("FILL", GlobalConstants.INT_FILLVAL, inplace=True)

    for time in events_dataset["EVENTTIMES"].data:
        index = np.where(events_dataset["EVENTTIMES"].data == time)[0][0]
        print(index)
        count = events_dataset["COUNT"].data[index]
        start_type = events_dataset["START_TYPE"].data[index]
        stop_type = events_dataset["STOP_TYPE"].data[index]
        start_position_tdc = events_dataset["START_POS_TDC"].data[index]

        if count == 0:
            assert df["Xf"].iloc[index] == GlobalConstants.INT_FILLVAL
            continue

        assert df["StartType"].iloc[index] == start_type
        assert df["StopType"].iloc[index] == stop_type

        # Check front x position
        # TODO: get correct lookup table
        xf = get_front_x_position(start_type, start_position_tdc)
        xf = float(df["Xf"].iloc[index])
        # assert xf == df['Xf'].iloc[index]

        # Check top and bottom
        if events_dataset["STOP_TYPE"].data[index] in [1, 2]:
            # Process for Top and Bottom stop types
            tof, particle_tof, (xb, yb) = get_back_positions(
                index, events_dataset, float(xf)
            )
            assert xb == float(df["Xb"].iloc[index])
            assert yb == float(df["Yb"].iloc[index])

            d, yf = get_front_y_position(start_type, yb)
            assert yf == pytest.approx(float(df["Yf"].iloc[index]), 1e-3)

            # pulse_height = events_dataset["ENERGY_PH"].data[index]
            # TODO: needs lookup table
            # energy = get_energy_pulse_height(pulse_height, stop_type, xb, yb)
            r = get_path_length((xf, yf), (xb, yb), d)
            assert r == pytest.approx(float(df["r"].iloc[index]), rel=1e-3)
            energy = float(df["Xf"].iloc[index])

            # TODO: needs lookup table to test bin
            ctof, bin = determine_species_pulse_height(energy, tof, r)
            assert ctof * 100 == pytest.approx(float(df["cTOF"].iloc[index]), rel=1e-3)

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
