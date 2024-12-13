
import numpy as np

from imap_processing import decom, imap_module_directory
from imap_processing.ultra.l0.decom_ultra import process_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
)
from imap_processing.ultra.l1a import ultra_l1a
from imap_processing.utils import group_by_apid


# these are copied from the conftest.py file in the ultra unit testing package
def ccsds_path_theta_0():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50_theta-0.00"
        "_20240207T102740.CCSDS"
    )
def xtce_path():
    """Returns the xtce image rates directory."""
    return (
        imap_module_directory
        / "ultra"
        / "packet_definitions"
        / "ULTRA_SCI_COMBINED.xml"
    )
#######
def de_dataset(ccsds=ccsds_path_theta_0(), xtce=xtce_path()):
    """L1A test data"""
    packets = decom.decom_packets(ccsds, xtce)
    grouped_data = group_by_apid(packets)
    decom_ultra_events = process_ultra_apids(
        grouped_data[ULTRA_EVENTS.apid[0]], ULTRA_EVENTS.apid[0]
    )
    decom_ultra_aux = process_ultra_apids(
        grouped_data[ULTRA_AUX.apid[0]], ULTRA_AUX.apid[0]
    )
    dataset = ultra_l1a.create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )
    # Remove start_type with fill values
    l1a_de_dataset = dataset.where(
        dataset["START_TYPE"] != np.iinfo(np.int64).min, drop=True
    )
    return l1a_de_dataset

