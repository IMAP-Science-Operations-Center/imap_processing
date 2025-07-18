"""Contains list of QFs to use for filtering."""

from imap_processing.quality_flags import (
    FlagNameMixin,
    ImapRatesUltraFlags,
)

QUALITY_FLAG_FILTERS: dict[str, list[FlagNameMixin]] = {
    "quality_attitude": [],
    "quality_ena_rates": [
        ImapRatesUltraFlags.FIRSTSPIN,
        ImapRatesUltraFlags.LASTSPIN,
    ],
}
