.. _idex-l3-scope:

L3 Scope - IDEX Does Not Have One Here
======================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Short page, because the answer is short: **there is no IDEX L3 in this
repository, and as far as the algorithm document and the code are concerned,
there is no IDEX L3 anywhere.**

Where IDEX stops
----------------

For most IMAP instruments, ``imap-processing`` stops at L2 and a separate
repository run closer to the science team produces L3. SWE's electron moments,
SWAPI's solar-wind proton parameters and IMAP-Lo's L3 maps all live outside this
repository, and those instruments' pages carry an "L3 scope" section warning you
off writing fitting or moment code here.

**IDEX is different.** It does not hand off at L2.

**[CODE]** ``imap_processing/__init__.py`` lists
``"idex": ["l1a", "l1b", "l2a", "l2b", "l2c"]`` - the highest level IDEX defines
is **L2C**, and L2C is already the mapped, aggregated, publication-shaped
product: monthly count-rate maps on a 6° rectangular ECLIPJ2000 grid. The
document's Figure 4.1 puts "Publication Level Products" immediately downstream of
L2B and L2C, as an output of the pipeline rather than another processing level.

**[DOC]** The acronym list in chapter 0 defines L0, L1A, L1B, L2A, L2B and L2C.
It does not define L3. No chapter mentions an IDEX L3 product, an L3 dependency,
or an L3 hand-off.

So the practical answer to "is IDEX L3 missing from this repository?" is: as far
as anything available here says, **no - it does not exist as a product level.**

.. note::

   This is a statement about the algorithm document and the code as of the
   1 June 2026 revision, not about the mission's Science Data Management Plan,
   which this repository does not contain. If the SDMP defines an IDEX L3, this
   page is the place to record it. Confirm with the IDEX team rather than
   assuming either way.

What the mapping infrastructure implies
---------------------------------------

L2C uses the **shared ENA-map machinery**
(``imap_processing/ena_maps/ena_maps.py``, ``AzElSkyGrid``, ``SkyTilingType``)
that IMAP-Lo, Hi and Ultra use for their sky maps. In those instruments that
machinery is L2-and-up territory; IDEX uses only the rectangular tiling and only
for binning counts, not for any of the ENA-specific pointing-set or
exposure-weighting logic.

``imap_processing/ena_maps/utils/naming.py`` treats IDEX (and GLOWS) as special
cases in two places - both instruments are excluded from parts of the standard
map-naming convention. If you are extending IDEX's map products, read those two
branches first.

What a real "L3" would need
---------------------------

If the IDEX team does eventually specify a higher-level product, the honest
statement of what L2 can hand it today is:

**Available and trustworthy:**

* Per-event impact charge, fitted pulse parameters and fit quality for
  ``Target_Low``, ``Target_High`` and ``Ion_Grid`` (L2A), masked where the
  channel saturated or the event was not a science event.
* Per-event velocity and mass estimates from all three low-rate channels (L2A),
  same masking. Note the mass unit question flagged in :ref:`idex-l2`.
* Per-event SPICE context: spacecraft position and velocity relative to the Sun,
  IDEX boresight longitude and latitude, solar longitude, spin phase (L1B,
  copied forward through L2A).
* Per-event classification and saturation flags (L1A, copied forward).
* Daily dust-hit counts and uptime-corrected rates by spin quadrant and by 6°
  sky pixel (L2B/L2C).

**Not available yet, and blocking any compositional science:**

* **The TOF mass spectrum.** ``mass``, ``mass_scale``, ``tof_snr``,
  ``tof_peak_kappa`` and every ``tof_peak_*`` variable are computed and then
  NaN-filled. Until the time-to-mass conversion and the EMG peak fits are
  validated, IDEX publishes no elemental composition at all - which is the
  instrument's headline measurement.
* **Mass- and charge-resolved rates.** ``counts_by_mass``, ``rate_by_mass``,
  ``counts_by_charge``, ``rate_by_charge`` and their map equivalents are
  fill-valued at L2B/L2C.
* **Combined-gain fits** for both the target pair and the TOF triple
  (**[DOC]** sections 4.7.8 and 4.7.10).

In other words, the gap in IDEX is not a missing processing level. It is that
the mass-spectrum half of the existing levels is written but withheld. See
:ref:`idex-implementation-status`.
