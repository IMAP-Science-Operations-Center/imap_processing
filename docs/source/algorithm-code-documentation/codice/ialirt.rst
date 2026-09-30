.. _codice-ialirt:

I-ALiRT
=======

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**[DOC]** Sections 10.4 and 14. **[CODE]**
``imap_processing/ialirt/l0/process_codice.py`` - the entire CoDICE I-ALiRT
algorithm lives in this one module, which reuses the ordinary L1A/L1B/L2
functions.

I-ALiRT (IMAP Active Link for Real-Time) is the < 5 minute latency space-weather
stream. It is **not** part of the ``imap_cli --instrument codice`` chain: it has
its own entry point and writes DynamoDB items rather than CDFs.

How the telemetry differs
-------------------------

**[DOC]** I-ALiRT data is packed **identically** to the regular science data but
transmitted differently: a trickle of small packets that must be reassembled.
Each packet has the CCSDS header plus:

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Mnemonic
     - Bits
     - Description
   * - ``SHCOARSE``
     - 32
     - Spacecraft time in seconds.
   * - ``ACQUISITION_TIME``
     - 32
     - Spacecraft time at the **end** of the acquisition cycle. All packets in a
       cycle share this.
   * - ``STATUS``
     - 8
     - Data quality / status.
   * - ``COUNTER``
     - 8
     - Which block of data this packet carries. 0-231 are valid; 231-239 are
       fill (0xFF).

then **15 data bytes** (Lo) or **5 data bytes** (Hi), a spare byte and a 16-bit
checksum.

**[DOC]** ``Plan ID``, ``Plan Step`` and ``View ID`` are all assumed to be 0, so
from the Views tab all I-ALiRT packets are **Lossy A + Lossless** compressed.

**[CODE]** ``COD_LO_COUNTER = 232``, ``COD_HI_COUNTER = 199``,
``COD_LO_RANGE = range(0, 15)``, ``COD_HI_RANGE = range(0, 5)``.
``find_groups`` (from ``ialirt/utils/grouping.py``) assembles complete counter
runs; ``concatenate_bytes`` stacks the ``cod_<sensor>_data_NN`` fields into a
single bytearray.

CoDICE-Lo I-ALiRT
-----------------

**[DOC]** Section 10.4.1. On board, data from all spin sectors and the five
sunward positions is summed into a single value, organised **by species** (not
by ESA step). Once a complete 0-231 set arrives it is processed **identically to
the SW_SPECIES_COUNTS product** with a single spin sector, a single azimuth and
128 energies. The only difference is the species list. Nominal species:

.. math::

   \mathrm{He^{++}},\ \mathrm{C^{+5}},\ \mathrm{C^{+6}},\ \mathrm{O^{+6}},\
   \mathrm{O^{+7}},\ \mathrm{O^{+8}},\ \mathrm{Mg},\ \mathrm{Fe(lowQ)},\
   \mathrm{Fe(hiQ)}

**[CODE]** ``LO_IALIRT_VARIABLE_NAMES``, with mass-per-charge
``LO_IALIRT_M_OVER_Q``:

.. list-table::
   :header-rows: 1
   :widths: 20 14 20 14 20 12

   * - Species
     - m/q
     - Species
     - m/q
     - Species
     - m/q
   * - ``heplusplus``
     - 2.0
     - ``oplus6``
     - 2.7
     - ``mg``
     - 3.5
   * - ``cplus5``
     - 2.4
     - ``oplus7``
     - 2.28
     - ``fe_hiq``
     - 3.85
   * - ``cplus6``
     - 2.0
     - ``oplus8``
     - 2.0
     - ``fe_loq``
     - 7.25

Pipeline
^^^^^^^^

**[CODE]** ``process_codice(dataset, l1a_lut_path, l2_lut_path, "codice_lo",
l2_geometric_factor_path)``:

1. ``find_groups`` -> ``concatenate_bytes`` -> one bytearray per cycle.
2. ``process_ialirt_data_streams`` splits the bit string by
   ``IAL_BIT_STRUCTURE`` (a 24-field dict mirroring the ordinary science packet
   header, including the post-2026-01-29 RGFO/NSO fields), then takes
   ``BYTE_COUNT`` bytes as the science payload. A packet with ``SHCOARSE == 0``
   is discarded.
3. ``create_xarray_dataset`` builds a **fake** dataset with an integer-index
   epoch and ``pkt_apid`` set to 1152 (Lo) or 1168 (Hi), so that the ordinary
   L1A functions can consume it.
4. ``process_by_table_id(cod_lo_dataset, l1a_lut_path, l1a_lo_species)`` - the
   **same** L1A function used for the science product, taking the
   ``COD_LO_IAL`` branch.
5. ``convert_to_rates(l1a_lo, "lo-ialirt")`` - the **same** L1B function.
   ``n_sectors`` = 12, except 11 at ESA step 127.
6. ``Logical_file_id`` is synthesised from the mid-measurement epoch because
   ``compute_geometric_factors`` parses the date out of it.
7. ``calculate_ratios`` -> the public products.

Pseudo-densities and ratios
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Section 14.2. First compute the intensity exactly as at L2:

.. math::

   J_j(l) = \frac{R_j(l)}{G_m \cdot \varepsilon_{jl1} \cdot (E/q)_l},
   \qquad
   U_j(l) = \frac{\sqrt{C_j(l)}}
   {G_m \cdot \varepsilon_{jl1} \cdot (E/q)_l \cdot (t_{acquire} \cdot 10^{-3} \cdot 12)}

Then form **pseudo**-densities. They are "pseudo" because several constant
factors that a real density needs are omitted - they cancel in the ratios:

.. math::

   d\_psN_j(l) = J_j(l) \cdot \sqrt{(E/q)_l} \cdot \sqrt{(m/q)_j},
   \qquad
   psN_j = \sum_{l=0}^{127} d\_psN_j(l)

.. math::

   psU_j = \sqrt{\sum_{l=0}^{127} \left(d\_psU_j(l)\right)^2}

Then the public products:

.. math::

   \frac{C}{O} &= \frac{psN_{C^{+5}} + psN_{C^{+6}}}
                       {psN_{O^{+6}} + psN_{O^{+7}} + psN_{O^{+8}}} \\
   \frac{Mg}{O} &= \frac{psN_{Mg}}
                        {psN_{O^{+6}} + psN_{O^{+7}} + psN_{O^{+8}}} \\
   \frac{Fe}{O} &= \frac{psN_{Fe,low} + psN_{Fe,high}}
                        {psN_{O^{+6}} + psN_{O^{+7}} + psN_{O^{+8}}} \\
   \frac{C^{+6}}{C^{+5}} &= \frac{psN_{C^{+6}}}{psN_{C^{+5}}}, \qquad
   \frac{O^{+7}}{O^{+6}} = \frac{psN_{O^{+7}}}{psN_{O^{+6}}}, \qquad
   \frac{Fe_{low}}{Fe_{high}} = \frac{psN_{Fe,low}}{psN_{Fe,high}}

**[DOC]** Ratio uncertainties propagate as

.. math::

   \Delta R = \frac{R_{top}}{R_{bot}}
   \sqrt{\left(\frac{\Delta R_{top}}{R_{top}}\right)^2
   + \left(\frac{\Delta R_{bot}}{R_{bot}}\right)^2}

with numerator and denominator uncertainties summed in quadrature first.

**[CODE]** ``calculate_ratios`` computes the six ratios, reusing
``get_geometric_factor_lut``, ``compute_geometric_factors``,
``get_efficiency_lut`` and ``process_lo_species_intensity`` from
``codice_l2.py`` with ``SOLAR_WIND_POSITIONS`` (position 1 only). Results are
rounded to six decimal places and wrapped in ``Decimal`` so DynamoDB can store
them. **Zero denominators return ``None`` rather than raising or producing
infinity** - the source notes this matches the instrument team's test data.

.. warning::

   **[CODE] The ratio uncertainties are not computed.** ``calculate_ratios``
   returns only the six ratios; there is no ``psU`` accumulation and no
   quadrature propagation. Section 14.2.2's uncertainty algorithm is unbuilt.

CoDICE-Hi I-ALiRT
-----------------

**[DOC]** Sections 10.4.2 and 14.1. On board, H counts are binned into **4 spin
sector bins** (summing every 6 of the 24 sectors), **4 azimuthal look
directions** (each a group of 3 SSDs) and **15 sqrt(2)-spaced energy-per-nucleon
bins**, accumulated over 4 spins at 1 min resolution.

SSD groups:

.. list-table::
   :header-rows: 1
   :widths: 16 30 30 24

   * - Group :math:`g`
     - SSD IDs
     - Elevation angle
     - Ref. spin angle :math:`\theta_{g,0}`
   * - 0
     - 0, 1, 3
     - 132.8 deg
     - 196.85 deg
   * - 1
     - 4, 5, 7
     - 65.7 deg
     - 174.55 deg
   * - 2
     - 8, 9, 11
     - 47.1 deg
     - 253.16 deg
   * - 3
     - 12, 13, 15
     - 114.3 deg
     - 275.44 deg

**[CODE]** ``HI_IALIRT_ELEVATION_ANGLE = [132.8, 65.7, 47.1, 114.3]`` matches.
``HI_IALIRT_REF_SPIN_ANGLE = [196.85, 174.55, 253.16, 275.44]`` **matches Rev 3
Chg 1**. The January 2026 draft printed values 90 deg higher (286.85, 264.55,
343.16, 5.44); see :ref:`codice-spin-angle-offset`. ``HI_IALIRT_SPIN_ANGLE`` in
``ialirt/utils/constants.py`` is built by adding 0, 90, 180 and 270 deg (mod
360) to each reference, matching **[DOC]** :math:`\theta_{g,n} = (\theta_{g,0}
+ 90^\circ n) \bmod 360^\circ`.

Rates and intensities
^^^^^^^^^^^^^^^^^^^^^

**[DOC]**

.. math::

   R(i, n, g) = \frac{C(i, n, g)}{6 \cdot 4 \cdot t_{acquire} \cdot 10^{-3}}
   \qquad [\mathrm{counts/s}]

.. math::

   I(i, n, g) = \frac{R(i, n, g)}{G_g \cdot \epsilon_{ig} \cdot \Delta E_i}
   \qquad [\#/(\mathrm{cm}^2\,\mathrm{sr}\,\mathrm{s}\,\mathrm{MeV/nuc})]

with :math:`G_g = 3 G_k = 0.039` cm2 sr (the sum over the three SSDs in the
group) and :math:`\epsilon_{ig}` the average of the H efficiencies for those
three SSDs.

**[CODE]** ``l1a_ialirt_hi`` produces the counts; ``convert_to_rates(l1a_hi,
"hi-ialirt")`` uses ``L1B_DATA_PRODUCT_CONFIGURATIONS["hi-ialirt"]`` =
``{num_spin_sectors: 6, num_spins: 4}`` times ``HI_ACQUISITION_TIME``.
``convert_to_intensities`` reads ``group_0`` ... ``group_3`` columns from the
efficiency CSV plus its ``GF`` row, and divides by ``g_g * eps_ig *
energy_passbands``. Output shape ``(4 spins, 15 energies, 4 spin sectors,
4 groups)``.

``IALIRT_HI_NUMBER_OF_SSD_PER_GROUP = 3.0`` exists in ``constants.py`` but the
grouping factor is expected to be baked into the ``GF`` row of the CSV.

Output
------

**[DOC]** Table 4 of Rev 3 Chg 1 lists both CoDICE I-ALiRT products (Hi H
intensities at 1 min; Lo C/O, Mg/O, Fe/O, C6+/C5+, O7+/O6+, Fe_loq/Fe_hiq) with
file prefix ``imap_ialirt_l1_realtime`` - the mission-wide I-ALiRT product -
rather than the ``imap_codice_l2-hi-ialirt_`` / ``imap_codice_l2-lo-ialirt_``
prefixes of the draft. That matches this repository: CoDICE writes no I-ALiRT
CDF of its own. Its fields go into the shared ``imap_ialirt_l1_realtime``
dataset built by ``ialirt/utils/create_xarray.py``. The synthesised
``Logical_file_id`` in step 6 above uses the same name.

**[CODE]** ``process_codice`` returns two lists of dicts, ready for DynamoDB:

.. code-block:: text

   codice_lo_data[i] = {
       ... instrument header items ...,
       "instrument": "codice_lo",
       "codice_lo_epoch": <int>,
       "codice_lo_c_over_o_abundance": Decimal | None,
       "codice_lo_mg_over_o_abundance": Decimal | None,
       "codice_lo_fe_over_o_abundance": Decimal | None,
       "codice_lo_c_plus_6_over_c_plus_5": Decimal | None,
       "codice_lo_o_plus_7_over_o_plus_6": Decimal | None,
       "codice_lo_fe_low_over_fe_high": Decimal | None,
   }

   codice_hi_data[i] = {
       ... instrument header items ...,
       "instrument": "codice_hi",
       "codice_hi_epoch": [<int>, ...],
       "codice_hi_h": <nested list of Decimal, shape (4, 15, 4, 4)>,
   }

Field names, dtypes and the Hi energy bin centres/deltas are declared in
``imap_processing/ialirt/utils/constants.py``
(``codice_hi_energy_center``, ``codice_hi_energy_minus``,
``codice_hi_energy_plus``); ``ialirt/utils/create_xarray.py`` treats
``codice_lo`` as a one-epoch instrument and ``codice_hi`` as multi-epoch.

.. note::

   ``process_codice``'s own docstring still says "This function is incomplete
   and will need to be updated". As of this survey the Lo ratio path and the Hi
   intensity path are both implemented end to end; the outstanding gap is
   uncertainty propagation.
