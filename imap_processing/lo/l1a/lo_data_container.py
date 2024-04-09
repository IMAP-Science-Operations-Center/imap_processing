"""Container for IMAP-Lo L0 data classes."""

from imap_processing.lo.l0.lo_apid import LoAPID


# TODO: Not sure if this class is worth having.
class LoContainer:
    """IMAP-Lo Level 0 data container class.

    The IMAP-Lo Level 0 data container class is used to store
    data for all 19 IMAP-Lo L0 unpacked data allow filtering
    by APID.

    Attributes
    ----------
    data : dict
        dictionary of IMAP-Lo data classes for L0


    Methods
    -------
    append():
        Appends another data class to the container

    filter_apid():
        Filters the container and returns a list of
        data classes for the APID
    """

    data_dict: dict

    def __init__(self):
        self.data_dict = {apid.value: [] for apid in LoAPID}

    # TODO: How do I write the parameter type in the docstring when it can be
    # multiple different kinds of types (any Lo dataclass)?
    def append(self, data):
        """
        Append data class to the container after confirming that the APID is valid.

        Parameters
        ----------
        data:
            Any Lo L0 data class object

        """
        data_apid = data.ccsds_header.PKT_APID
        # check if the data being appended has a valid Lo APID
        if data_apid in [apid.value for apid in LoAPID]:
            self.data_dict[data_apid].append(data)

    def filter_apid(self, apid):
        """
        Filter the container by APID.

        Parameters
        ----------
        apid: int
            APID to use for data filtering.

        Returns
        -------
        list
            filtered list of Lo data classes for a single APID.
        """
        return self.data_dict[apid]
