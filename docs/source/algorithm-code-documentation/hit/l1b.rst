.. _hit-l1b:

L1A to L1B: Livetime Correction and Rates
=========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**[DOC]** Section 6 of the algorithm document.

* **Input**: L1A CDF files (science), or the L0 CCSDS file again (housekeeping)
* **Processing requirements**: instrument livetime, conversion tables for
  housekeeping
* **Output**: L1B CDF files

L1B is conceptually the simplest level: **divide counts by the fractional
livetime**. Two wrinkles make it more than that - the livetime counter needs a
nonlinear unpacking, and the sectored rates need livetime from a *different*
10 minutes.

.. _hit-l1b-livetime:

The livetime fraction
---------------------

**[DOC]** Section 6.2. The livetime counter (``ERATES[0]``, frame bytes 6-7)
counts **16 MHz FPGA clock cycles during which the front-end electronics were
waiting for a trigger**. A full 60-second frame at 100% livetime would be
**960,000,000 cycles**.

That does not fit in the 16-bit compressed field. The document is explicit:
for nominal 60-second frames, **any livetime of 14% or more overflows and the
field rolls over**. Rather than fix the encoding, the conversion is defined as
a **three-segment piecewise linear function of the decompressed counter**,
valid from about 0.002% to 110% livetime:

.. math::

   \mathrm{Livetime\ Fraction} =
   \begin{cases}
     \mathrm{LIVE\_TIME} \times 1.04 \times 10^{-9}
       & \mathrm{LIVE\_TIME} > 16000 \\
     \mathrm{LIVE\_TIME} \times 3.41 \times 10^{-5} + 0.14
       & 0 \le \mathrm{LIVE\_TIME} \le 4101 \\
     \mathrm{LIVE\_TIME} \times 6.827 \times 10^{-5}
       & 4101 \le \mathrm{LIVE\_TIME} \le 16000
   \end{cases}

Note the ``+ 0.14`` offset on the first segment - that is the rollover being
undone. Note also that the segments are written in a deliberately confusing
order in the document, and that 4101 and 16000 appear in two branches each.

.. warning::

   **[DOC]** The conversion is **wrong below LIVE_TIME = 16000 corresponding to
   0.001667% livetime**. The document says this situation "should be detectable
   via other mnemonics, such as ``NUMTRIG`` (Table 12) or STIM event counts
   (``PBUFRATES`` #29 and #30, Table 16)". **No such detection exists in the
   code** - there is no quality flag for an out-of-range livetime. See
   :ref:`hit-gap-livetime-range`.

**[CODE]** ``hit_l1b.livetime_fraction_calculation`` implements exactly this,
resolving the overlapping bounds as ``<= 4101``, ``> 4101 and <= 16000``,
``> 16000``:

.. code-block:: python

   livetime1 = livetime_counter <= 4101
   livetime2 = (livetime_counter > 4101) & (livetime_counter <= 16000)
   livetime3 = livetime_counter > 16000

   livetime_fraction[livetime1] = livetime_counter[livetime1] * 3.41e-5 + 0.14
   livetime_fraction[livetime2] = livetime_counter[livetime2] * 6.827e-5
   livetime_fraction[livetime3] = livetime_counter[livetime3] * 1.04e-9

.. note::

   ``l1b/constants.py`` defines ``LIVESTIM_PULSES = 270`` with the comment
   "Expected number of livestim pulses per integration time. This is used to
   calculate the fractional livetime". **It is not used anywhere.** It is a
   remnant of an earlier approach (the document's section 4.3.2.2 mentions
   using livestim data to characterise the livetime error). Treat it as dead.

Counts to rates
---------------

**[DOC]** Equation 9:

.. math::

   \mathrm{Count\ Rate} = \frac{\mathrm{Counts}}{\mathrm{Livetime\ Fraction}}

Applied to **all** science rates (document Tables 13, 15-24). The units stay
"counts per integration time" - the values are simply no longer integers. The
uncertainties are divided by the same livetime.

**[CODE]** ``hit_l1b.calculate_rates`` divides the value and both uncertainty
arrays by ``livetime`` and casts to ``float32``.

L1B standard rates
------------------

**[CODE]** ``process_standard_rates_data``. It copies these twelve arrays from
L1A, along with their two uncertainty companions each, and divides all of them
by livetime:

.. code-block:: text

   sngrates     coinrates    pbufrates
   l2fgrates    l2bgrates
   l3fgrates    l3bgrates
   penfgrates   penbgrates
   ialirtrates  l4fgrates    l4bgrates

Plus ``dynamic_threshold_state`` (renamed from ``hdr_dynamic_threshold_state``),
which L2 needs.

Deliberately **not** included: ``sectorates`` (it becomes its own product) and
the event-processing counters ``nread`` .. ``nbadtags`` and the ``num_*``
hazard/trigger counters from ERATES.

.. note::

   **[CODE]** ``initialize_l1b_dataset`` is given an explicit coordinate list
   that omits ``l4fgrates_index`` and ``l4bgrates_index``. The dimensions get
   created implicitly when the data arrays are assigned, so this works, but the
   two L4 arrays end up without explicit index coordinates. Harmless today;
   worth knowing if you add anything that keys off those coordinates.

.. _hit-l1b-summed:

L1B summed rates
----------------

**[DOC]** Section 6.3 and Table 28. Standard-rate energy bins are combined into
**wider bins with better counting statistics**, useful during quiet times. Two
things happen at once:

* **Bins are merged across penetration ranges.** A single summed bin draws
  from R2, R3 and R4 counters simultaneously. E.g. "H 1.8-3.6 MeV/nuc" sums
  four R2 bins and two R3 bins.
* **Species are merged into groups.** ``he`` = He-3 + He-4 across all its
  contributing bins. (The document also names CNO and NeMgSi as groups in this
  section, but Table 28 does not actually define them - they only exist in the
  sectored product.)

Then divide by livetime.

**[DOC]** *"The Lev1B Summed Rate uncertainties are calculated by summing the
upper and lower uncertainties from Lev1A and dividing by the livetime, just as
is done in calculating the Lev1B science variables."*

.. important::

   That is a **linear** sum of uncertainties, not a quadrature sum. It is what
   the document specifies and what the code does. It is conservative (it
   overestimates the combined uncertainty for independent bins). Do not
   "correct" it to quadrature without asking the HIT team - and note that the
   same linear summing is then used again at L2 for the standard intensity
   product.

**[CODE]** ``SUMMED_PARTICLE_ENERGY_RANGE_MAPPING`` in ``l1b/constants.py`` is
the machine-readable form of Table 28: for each species, a list of
``{"energy_min", "energy_max", "R2": [...], "R3": [...], "R4": [...]}`` where
the lists are **indices into ``l2fgrates`` / ``l3fgrates`` / ``penfgrates``**
(i.e. Particle IDs).

.. list-table::
   :header-rows: 1
   :widths: 20 16 64

   * - Species
     - Bins
     - Energy bins (MeV/nuc)
   * - ``h``
     - 4
     - 1.8-3.6, 4-6, 6-10, 10-15
   * - ``he3``
     - 3
     - 4-6, 6-10, 10-15
   * - ``he4``
     - 4
     - 1.8-3.6, 4-6, 6-10, 10-15
   * - ``he``
     - 3
     - 4-6, 6-10, 10-15 (He-3 + He-4)
   * - ``c``, ``n``, ``o``, ``ne``, ``mg``
     - 4 each
     - 4-6, 6-10, 10-15, 15-27
   * - ``na``
     - 2
     - 10-15, 15-27
   * - ``al``, ``ni``
     - 3 each
     - 6-10 (Al) / 10-15, 15-27, 27-40
   * - ``si``, ``s``, ``ar``, ``ca``, ``fe``
     - 5 each
     - 4-6, 6-10, 10-15, 15-27, 27-40
   * - **Total**
     - **67**
     - Matches the 67 rows in each ``imap_hit_summed-dt<N>-factors`` CSV.

The mechanics live in ``hit_utils``:

* ``initialize_particle_data_arrays`` creates zero-filled
  ``(epoch, n_bins)`` arrays for the species and its two uncertainties.
* ``sum_particle_data`` does the actual
  ``l2fgrates[:, R2].sum(axis=1) + l3fgrates[:, R3].sum(axis=1) +
  penfgrates[:, R4].sum(axis=1)``, and the same for both uncertainty arrays.
* ``add_energy_variables`` writes ``<species>_energy_mean`` and the two
  deltas.
* ``add_summed_particle_data_to_dataset`` orchestrates the three.

.. important::

   **The same three helpers are reused at L2** to build the *standard*
   intensity product from L1B standard rates. The only difference is which
   mapping dict is passed in. If you change ``sum_particle_data`` you change
   both products.

.. _hit-l1b-energy-mean:

Energy bin identification - a real deviation
--------------------------------------------

**[DOC]** Section 6.1 is unambiguous. At every level above L1A, an energy bin
is identified by its **geometric mean** and its edges:

.. math::

   E_{char} = \sqrt{E_{min} \cdot E_{max}}

.. math::

   \mathrm{delta\_plus} = E_{max} - E_{char}, \qquad
   \mathrm{delta\_minus} = E_{char} - E_{min}

The document explains why: the true mean energy of particles in a bin varies
with the spectrum (even during a single event), so the geometric mean is only
a **characteristic** label. The bin **limits** are the fundamental quantity.

**[CODE]** ``hit_utils.add_energy_variables`` uses the **arithmetic** mean:

.. code-block:: python

   energy_mean = np.round(
       np.mean(np.array([energy_min_values, energy_max_values]), axis=0), 3
   ).astype(np.float32)

This affects ``<species>_energy_mean``, ``_energy_delta_plus`` and
``_energy_delta_minus`` in **every** L1A sectored, L1B summed, L1B sectored,
L2 standard, L2 summed and L2 macropixel product. It is the most widespread
doc/code deviation in HIT. See :ref:`hit-gap-geometric-mean`.

.. _hit-l1b-sectored:

L1B sectored rates
------------------

**[DOC]** Sections 4.2.2 and 6.2. Three things make this different from every
other rate:

#. **The integration time is 10 minutes, not 1.** A complete sectored set is 10
   consecutive frames.
#. **Counts are transmitted 10 minutes after they are collected.** Block *n*'s
   counts must be divided by block *n-1*'s livetime. The document's Figure 14
   is a timeline of exactly this.
#. **A factor of 15.** Each look direction only sees its inclination bin for
   1/15 of a rotation, so the raw counts are divided by 15 as well.

.. math::

   \mathrm{Livetime}_{sector} = \sum_{i=0}^{9} \mathrm{Livetime\ Fraction}_i

.. math::

   \mathrm{Count\ Rate}_{sector} =
   \frac{\mathrm{Raw\ Counts}}{15 \times \mathrm{Livetime}_{sector}}

**[CODE]** The 10-minute *shift* is done at L1A (``subset_livetime``, see
:ref:`hit-l1a`); L1B only has to do the *sum* and the division.

* ``sum_livetime_10min`` sums the livetime fraction in non-overlapping
  10-element windows and ``np.repeat``\ s each sum 10 times, so the result has
  the same shape as the input:
  ``[5,5,5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6,6,6, ...]``.
* ``process_sectored_rates_data`` then computes, for each
  ``<species>_sectored_counts`` array:

  .. code-block:: python

     rates = np.where(
         counts != FILLVAL_INT64,
         (counts / (SECTORS * livetime_10min_reshaped)).astype(np.float32),
         FILLVAL_FLOAT32,
     )

  with ``SECTORS = 15`` and livetime reshaped to ``[:, None, None, None]`` to
  broadcast over ``(energy, azimuth, zenith)``.
* The variables are renamed from ``<species>_sectored_counts`` to plain
  ``<species>``.

Two notes on the implementation:

* It drops out of xarray into NumPy deliberately - the comment explains that
  the counts and the livetime live on **different epoch coordinates**
  (``epoch`` vs ``epoch_livetime``), so xarray's automatic alignment would
  otherwise produce an empty result.
* The fill value **changes type** here: int64 fill in, float32 fill
  (``-1.00e31``) out. Both are defined in ``l1b/constants.py``.

.. _hit-l1b-hk:

L1B housekeeping
----------------

**[DOC]** Section 6.5, Tables 29 and 30. Raw DN are converted to volts and
temperatures:

* **Preamp voltages** and the EBOX supply rails: simple linear
  ``V = a * DN`` or ``V = a * DN + b``. E.g.
  ``+5.7VA Ebox: V = 0.001835 * DN``,
  ``-12VA Ebox: V = 0.004680 * DN - 13.95``,
  ``L3/4A Bias: V = 0.06712 * DN - 2.77``.
* **Temperatures** (``TEMP0``-``TEMP3``, ``ANALOG_TEMP``, ``HVPS_TEMP``,
  ``IDPU_TEMP``, ``LVPS_TEMP``): thermistor lookups from
  HIT-ELEC-HDBK-0008, reproduced as Table 30: 191 rows covering
  **-40 to +150 degrees C in 1-degree steps**, each giving the thermistor
  resistance and the expected voltage and DN for both the Analog board and
  the FEE board.
* Everything else is passed through unchanged.

.. note::

   Table 29 has a copy-paste error: the ``Preamp L1A``, ``Preamp L1B`` and
   ``Preamp L234B`` rows all give the equation as
   ``V = 0.00121 * Digital L234A Preamp``, naming the L234A input for all
   four. The intent is clearly ``V = 0.00121 * <that channel's own DN>``.

**[CODE]** **None of this is in Python.** The conversions live in the XTCE at
``imap_processing/hit/packet_definitions/hit_packet_definitions.xml`` as
``PolynomialCalibrator`` elements (156 of them), with the eight thermistor
channels using ``ContextCalibrator`` chains (20-22 context ranges each) to
express the piecewise lookup.

``hit_l1b`` therefore does not compute anything for housekeeping - it just
re-reads the **L0 CCSDS file** with ``derived=True``:

.. code-block:: python

   datasets_by_apid = get_datasets_by_apid(packet_file, derived=True)
   l1b_dataset = process_housekeeping_data(
       datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1b_hk"
   )

That is why the L1B housekeeping CLI branch takes an **L0** dependency, not an
L1A one. If a conversion is wrong, fix the XTCE.
