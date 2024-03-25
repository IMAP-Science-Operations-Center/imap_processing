"""Parse Lo Level 0 data class attributes."""


def set_attributes(dataclass, packet):
    """Set Lo Level 0 data class attributes using packet data.

    Parameters
    ----------
    packet : dict
        A single Lo L0 packet from space packet parser.

    """
    for key, item in packet.items():
        if hasattr(dataclass, key):
            value = (
                item.derived_value if item.derived_value is not None else item.raw_value
            )
            setattr(dataclass, key, value)
        else:
            raise KeyError(
                f"Did not find matching attribute in {dataclass.__class__} data class"
                f"for {key}"
            )
