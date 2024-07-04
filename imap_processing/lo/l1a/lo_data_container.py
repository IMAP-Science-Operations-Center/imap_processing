"""Container for IMAP-Lo L0 data classes."""

from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l0.utils.lo_base import LoBase


# TODO: Not sure if this class is worth having.
class LoContainer:
    """
    IMAP-Lo Level 0 data container class.

    The IMAP-Lo Level 0 data container class is used to store
    data for all 19 IMAP-Lo L0 unpacked data to allow filtering
    by APID.

    Attributes
    ----------
    data : dict
        Dictionary of IMAP-Lo data classes for L0.

    Methods
    -------
    add():
        Adds another data class to the container.
    filter_apid():
        Filters the container and returns a list of
        data classes for the APID.
    """

    data_dict: dict

    def __init__(self) -> None:
        self.data_dict = {apid.value: [] for apid in LoAPID}

    # TODO: How do I write the parameter type in the docstring when it can be
    # multiple different kinds of types (any Lo dataclass)?
    def add(self, data: LoBase) -> None:
        """
        Add data class to the container after confirming that the APID is valid.

        Parameters
        ----------
        data : LoBase
            Any Lo L0 data class object.
        """
        data_apid = data.ccsds_header.PKT_APID
        # check if the data being appended has a valid Lo APID
        if data_apid in [apid.value for apid in LoAPID]:
            self.data_dict[data_apid].append(data)

    def filter_apid(self, apid: int) -> list:
        """
        Filter the container by APID.

        Parameters
        ----------
        apid : int
            APID to use for data filtering.

        Returns
        -------
        list
            Filtered list of Lo data classes for a single APID.
        """
        filtered_list: list = self.data_dict[apid]
        return filtered_list
