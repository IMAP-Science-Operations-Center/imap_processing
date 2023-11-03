from pathlib import Path

from imap_processing import imap_module_directory
from imap_processing.cdf_utils import write_cdf
from imap_processing.swe import __version__, swe_cdf_attrs
from imap_processing.swe.l1b.swe_l1b_science import swe_l1b_science
from imap_processing.swe.utils.swe_utils import SWEAPID, filename_descriptors
from imap_processing.utils import convert_raw_to_eu


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

    if apid == SWEAPID.SWE_SCIENCE:
        data = swe_l1b_science(eu_data)
    else:
        data = eu_data
        data.attrs.update(swe_cdf_attrs.l1b_science_attrs)
        # TODO: find out why logical source was coming out
        # as below at this point in hk processing
        # {
        #     "logical_source": ["imap_swe_l1"]
        # }
        data.attrs["Logical_source"] = "imap_swe_l1b"
        data.attrs["Data_version"] = __version__
    return write_cdf(
        data,
        mode=f"{data['APP_MODE'].data[0]}" if apid == SWEAPID.SWE_APP_HK else "",
        description=filename_descriptors.get(apid),
        directory=current_dir,
    )
