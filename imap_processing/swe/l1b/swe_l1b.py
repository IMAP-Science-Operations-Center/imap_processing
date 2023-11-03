from pathlib import Path

from imap_processing import imap_module_directory
from imap_processing.swe import __version__
from imap_processing.swe.l1b.swe_l1b_science import swe_l1b_science
from imap_processing.swe.utils.swe_utils import SWEAPID, filename_descriptors
from imap_processing.utils import convert_raw_to_eu
from imap_processing.write_to_cdf import write_to_cdf


def swe_l1b(l1a_dataset):
    """Process data to L1B.

    Parameters
    ----------
    l1a_dataset : xr.dataset
        l1a data input
    """
    apid = l1a_dataset["PKT_APID"].data[0]
    current_dir = Path(__file__).parent

    # convert value from raw to engineering units as needed
    conversion_table_path = (
        f"{imap_module_directory}/swe/l1b/engineering_unit_convert_table.csv"
    )
    # Look up packet name from APID
    packet_name = next(packet for packet in SWEAPID if packet.value == apid)
    # Convert raw data to engineering units as needed
    eu_data = convert_raw_to_eu(
        l1a_dataset,
        conversion_table_path=conversion_table_path,
        packet_name=packet_name.name,
    )

    if apid == SWEAPID.SWE_APP_HK.value:
        # Save to cdf
        return write_to_cdf(
            eu_data,
            "swe",
            "l1b",
            version=__version__,
            mode=f"{eu_data['APP_MODE'].data[0]}",
            description=filename_descriptors.get(apid),
            directory=current_dir,
        )
    elif apid == SWEAPID.SWE_SCIENCE.value:
        data = swe_l1b_science(eu_data)
        return write_to_cdf(
            data,
            "swe",
            "l1b",
            version=__version__,
            description=filename_descriptors.get(apid),
            directory=current_dir,
        )
    else:
        # Don't process other data further
        # TODO log it
        return None
