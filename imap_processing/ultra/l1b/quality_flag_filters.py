"""Contains list of QFs to use for filtering."""

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags

QUALITY_FLAG_FILTERS = {
    "quality_attitude": [ImapAttitudeUltraFlags.SPINRATE],
    "quality_ena_rates": [ImapRatesUltraFlags.HIGHRATES],
}
