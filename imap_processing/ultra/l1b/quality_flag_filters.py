"""Contains list of QFs to use for filtering."""

from imap_processing.quality_flags import (
    FlagNameMixin,
    ImapDEOutliersUltraFlags,
    ImapDEScatteringUltraFlags,
    ImapRatesUltraFlags,
)

SPIN_QUALITY_FLAG_FILTERS: dict[str, list[FlagNameMixin]] = {
    "quality_attitude": [],
    "quality_ena_rates": [
        ImapRatesUltraFlags.FIRSTSPIN,
        ImapRatesUltraFlags.LASTSPIN,
    ],
}

DE_QUALITY_FLAG_FILTERS: dict[str, list[FlagNameMixin]] = {
    "quality_outliers": [
        ImapDEOutliersUltraFlags.FOV,
        ImapDEOutliersUltraFlags.DURINGREPOINT,
        ImapDEOutliersUltraFlags.COINPH,
        ImapDEOutliersUltraFlags.BACKTOF,
    ],
    "quality_scattering": [
        ImapDEScatteringUltraFlags.ABOVE_THRESHOLD,
    ],
}
