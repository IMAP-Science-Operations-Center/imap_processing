"""Bitwise flagging."""

from enum import IntFlag


class L1bQualityFlags(IntFlag):
    """Quality flags."""

    NONE = 0x0

    BASE_INF = 0x1  # bit 0, Infinite value
    BASE_MISSING_TELEM = 0x2  # bit 1, Missing telemetry
    BASE_NEG = 0x4  # bit 2, Negative value
    BASE_RES1 = 0x8  # bit 3, Reserved 1

    # ENA L1b specific flags
    ENA_RES1 = 0x10  # bit 4, ENA reserved 1

    # Ultra L1b specific flags
    ULTRA_RES1 = 0x20  # bit 5, Ultra reserved 1
    ULTRA_RES2 = 0x40  # bit 6, Ultra reserved 2

    # Shorthands for mixed states.
    ALL = (
        BASE_INF
        | BASE_MISSING_TELEM
        | BASE_NEG
        | BASE_RES1
        | ENA_RES1
        | ULTRA_RES1
        | ULTRA_RES2
    )
    BASE_ALL = BASE_INF | BASE_MISSING_TELEM | BASE_NEG | BASE_RES1

    def __str__(self) -> str:
        """
        Override the default string representation.

        Returns
        -------
        output_string : str
            The modified string representation without the initial segment
            before the first period.
        """
        output_string = super().__str__().split(".", 1)[-1]

        return output_string
