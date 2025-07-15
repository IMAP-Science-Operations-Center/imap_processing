"""Contains list of QFs to use for filtering."""

from imap_processing.quality_flags import (
    FlagNameMixin,
    ImapAttitudeUltraFlags,
    ImapRatesUltraFlags,
)

QUALITY_FLAG_FILTERS: dict[str, list[FlagNameMixin]] = {
    "quality_attitude": [
        ImapAttitudeUltraFlags.SPINRATE,
        ImapAttitudeUltraFlags.AUXMISMATCH,
    ],
    "quality_ena_rates": [
        ImapRatesUltraFlags.HIGHRATES,
        ImapRatesUltraFlags.FIRSTSPIN,
        ImapRatesUltraFlags.LASTSPIN,
    ],
}
