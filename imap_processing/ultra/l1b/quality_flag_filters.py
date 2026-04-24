"""Contains list of QFs to use for filtering."""

from imap_processing.quality_flags import (
    FlagNameMixin,
    ImapDEOutliersUltraFlags,
    ImapDEScatteringUltraFlags,
    ImapRatesUltraFlags,
)

SPIN_QUALITY_FLAG_FILTERS: dict[str, list[FlagNameMixin]] = {
    "quality_attitude": [],  # This is empty for now but can be populated with attitude
    # flags in the future
    "quality_ena_rates": [
        ImapRatesUltraFlags.FIRSTSPIN,
        ImapRatesUltraFlags.LASTSPIN,
    ],
}
# The following quality flag arrays contain flags that are dynamically created
# In ULTRA l1b extended spin. The flags are created based on the number
# Of bins that were used to group energies. If the flag array is in this list,
# Then all flags in the array will be used for filtering.
ENERGY_DEPENDENT_SPIN_QUALITY_FLAG_FILTERS: list = [
    "quality_low_voltage",
    "quality_upstream_ion_1",
    "quality_upstream_ion_2",
    "quality_spectral",
    "quality_high_energy",
    "quality_statistics",
]

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
