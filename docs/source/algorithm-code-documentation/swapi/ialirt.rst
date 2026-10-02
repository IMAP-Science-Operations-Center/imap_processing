.. _swapi-ialirt:

I-ALiRT - Real-Time Space Weather Stream
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Code:** ``imap_processing/ialirt/l0/process_swapi.py`` (368 lines, the whole
algorithm), ``imap_processing/ialirt/constants.py``
(``class IalirtSwapiConstants``),
``imap_processing/ialirt/packet_definitions/ialirt_swapi.xml`` (APID 1187).

**Document:** section 14.

.. note::

   The SWAPI I-ALiRT code does **not** live under ``imap_processing/swapi/``.
   It lives with the other instruments' I-ALiRT parsers and imports two things
   from the SWAPI modules: ``process_sweep_data`` (the 6-per-packet to
   72-per-sweep reorder) and ``SWAPI_LIVETIME``. Its output is a **list of
   dicts destined for the I-ALiRT database, not a CDF**, and there is no
   production caller in this repository - ``process_swapi_ialirt`` is invoked
   by the SDC's real-time service.

What it is
----------

**[DOC]** The I-ALiRT cadence requirement for instruments is 15 seconds. A
single SWAPI sweep is 12 seconds, so SWAPI delivers **one record per sweep at a
12-second cadence**, which comfortably beats the requirement.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Property
     - Value
   * - Data products
     - Pseudo speed, density and temperature of H\ :sup:`+` solar wind
   * - Time resolution
     - 12 s
   * - Components
     - :math:`u_p`, :math:`n_p`, :math:`T_p` (isotropic)
   * - Inputs required
     - Coincidence count rates and count rate uncertainties as a function of
       E/q for **62 coarse energy bins (1 sweep)**
   * - Ancillary files
     - Effective area :math:`A_{\mathrm{eff}}`, FWHM of energy width
       :math:`\Delta E/E`

The word **pseudo** is doing real work here. These are not the L3a moments.
They come from an analytical model with no angular response function, fit to a
handful of bins around the peak, on a single sweep. They are meant for space
weather forecasting, not science.

The packet
----------

**[CODE]** ``ialirt_swapi.xml``, APID **1187**, fields:

.. code-block:: text

   SWAPI_SHCOARSE      packet time
   SWAPI_ACQ           acquisition time (used for the record's midpoint)
   SWAPI_FLAG
   SWAPI_RESERVED
   SWAPI_SEQ_NUMBER    0-11, as in SWP_SCI
   SWAPI_VERSION       used as the sweep table id for the LUT lookup
   SWAPI_COIN_CNT0..5  six coincidence counts, one per 1/6 s
   SWAPI_SPARE

**Coincidence counts only** - no PCEM, no SCEM, no per-sample range-status
bits. This is the subset the document describes.

The analytical count-rate model
-------------------------------

**[DOC]** Originally developed for SWAP (Elliott et al. 2016). It uses the
instrument effective area :math:`A_{\mathrm{eff}}`, the azimuthal width of the
field of view :math:`\Delta\phi`, and the energy passband :math:`E_e`, and
**does not require any angular response function**. The count rate for each ESA
energy passband is

.. math::

   C(E_e) = \left( n A_{\mathrm{eff}}
     \left(\frac{\beta}{\pi}\right)^{3/2}
     e^{-\beta\left(v_e^2 + u^2 - 2 v_e u\right)} \right)
     \sqrt{\frac{\pi}{\beta u v_e}}\;
     \mathrm{erf}\!\left(\sqrt{\beta u v_e}\,\frac{\Delta\phi}{2}\right)
     \left( v_e^4 \frac{\Delta v}{v_e}
     \sin^{-1}\!\left(v_{\mathrm{th}}/v_e\right) \right)

where :math:`n` is the solar wind density, :math:`u` the bulk flow speed,
:math:`v_e = \sqrt{2 E_e/m}` the centre speed of the passband,
:math:`\Delta v` the speed width of the passband,
:math:`v_{\mathrm{th}} = \sqrt{2 k_B T/m}` the thermal velocity, and
:math:`\beta = 1/v_{\mathrm{th}}^2`.

The speed width comes from the FWHM of the energy width:

.. math::

   \frac{\Delta v}{v} = \frac{1}{2}\left(\frac{\Delta E}{E}\right)

**[CODE]** ``count_rate(energy_pass, speed, density, temp)`` implements this
term for term, with unit conversions (energy eV to J, speed km/s to m/s,
density cm\ :sup:`-3` to m\ :sup:`-3`).

Constants
---------

**[CODE]** ``IalirtSwapiConstants``:

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Constant
     - Value
     - Source
   * - ``eff_area``
     - :math:`1.633 \times 10^{-4}` cm\ :sup:`2`
     - **[DOC]** matches
   * - ``az_fov``
     - 30 degrees (as radians)
     - **[DOC]** matches
   * - ``fwhm_width``
     - 0.085
     - **[DOC]** matches ("SWAPI's energy resolution for solar wind protons")
   * - ``speed_ew``
     - :math:`0.5 \times 0.085`
     - **[DOC]** matches
   * - ``boltz``, ``at_mass``, ``prot_mass``, ``e_charge``
     - SI
     - --
   * - ``speed_coeff``
     - :math:`\sqrt{2 e/m_p}/10^3`
     - Used for the initial speed guess from the peak energy
   * - ``temporary_density_factor``
     - :math:`e^1`
     - **[DOC]** the flight-data density correction, see below

The flight-data corrections
---------------------------

**[DOC]** Section 14, "Processing update for flight data". Two corrections were
added after first light:

1. **Spin-phase smearing.** "The solar wind proton parameters appear to show
   significant change within 5 sweeps due to S/C spin phase variation. A simple
   correction using 5 sweeps averaged (2 sweeps before, 2 sweeps after, and 1
   current) values is provided."
2. **Density offset.** "The count rates in the space appear to be much higher
   compared to those shown in Figure 24, which made the pseudo density much
   higher compared to a realistic value. This systematic offset is accounted for
   by correcting the pseudo density by scaling it to :math:`1/e` times the
   5-sweep-averaged fitted density."

**[CODE]** Both are implemented, the second exactly and the first with a
difference:

* The density correction is applied *inside the model* rather than to the
  output: ``density = density * exp(1)`` in ``count_rate`` inflates the model
  count by :math:`e`, which makes the fitted density come out :math:`e^{-1}`
  times smaller. Algebraically equivalent, and the comment says so - "this will
  increase the model count by a factor of :math:`e^1`, changing the output
  density by a factor of :math:`e^{-1}` ... to be replaced once SWAPI's L3
  processing pipeline is finalized."
* The 5-sweep average is a **trailing** window (``[-5:]``: the current sweep
  plus the 4 before it), not the document's **centred** window (2 before, the
  current, 2 after). It is also a **geometric** mean, which the document does
  not specify. See :ref:`swapi-implementation-status`.

The algorithm as implemented
----------------------------

**[CODE]** ``process_swapi_ialirt(unpacked_data, calibration_lut_table)``:

.. code-block:: text

   1. sort by epoch; compute MET from sc_sclk_sec / sc_sclk_sub_sec
   2. find_groups(..., (0, 11), "swapi_seq_number", "met")     group into sweeps
   3. per group:
      a. require seq numbers to be exactly [0..11], else skip the group
      b. process_sweep_data(subset, "swapi_coin_cnt")          -> 72 values
      c. counts = counts * 16 + 8                              decompression
      d. truncate to the first 63 steps
      e. rate  = counts / SWAPI_LIVETIME
         error = sqrt(counts) / SWAPI_LIVETIME
      f. look up the 63 energies from the esa-unit-conversion table, using
         swapi_version as the sweep id and the LATEST timestamp
      g. optimize_pseudo_parameters(rate, error, energies)
      h. once 5 consecutive sweeps (12.0 +/- 0.05 s apart) are in hand,
         geometric-mean the last 5 and emit one record

Grouping differs from the science pipeline: ``find_groups`` plus an explicit
``np.array_equal(seq_values, np.arange(12))`` check, rather than the
timestamp-difference sliding window used at L1. Incomplete or duplicated groups
are collected and logged, not raised.

Decompression
^^^^^^^^^^^^^

**[CODE]**

.. code-block:: python

   # I-ALiRT packets have counts compressed by a factor of 16.
   # Add 8 to avoid having counts truncated to 0 and to avoid
   # counts being systematically too low
   raw_coin_count = raw_coin_count * 16 + 8

Unconditional - there are no range-status bits in the I-ALiRT packet, so every
count is assumed compressed. The ``+ 8`` (half of 16) is a mid-bin correction
for the truncation the on-board divide introduces. **Neither the unconditional
multiply nor the ``+8`` appears in the document.**

The fit
^^^^^^^

**[DOC]** The pseudo speed, density and temperature are found by minimizing the
sum of **inverse-variance weighted** squares of the difference between modelled
and observed count rates, using a non-linear least squares algorithm (e.g.
Levenberg-Marquardt). "The energy range considered for the analytical model fit
are **three energy bins on the left and two on the right** of
:math:`E_{\mathrm{peak}}`", where :math:`E_{\mathrm{peak}}` is the energy of
the ESA bin with the highest count rate.

**[CODE]** ``optimize_pseudo_parameters``:

.. code-block:: python

   max_index = np.argmax(count_rates)
   five_point_range = range(max_index - 2, max_index + 2 + 1)
   xdata = energy_passbands.take(five_point_range, mode="clip")
   ...
   curve_fit(f=count_rate, xdata=xdata, ydata=ydata, sigma=sigma, p0=initial_param_guess)

``scipy.optimize.curve_fit`` with ``sigma`` is inverse-variance weighted
least squares (Levenberg-Marquardt by default for unbounded problems), so the
minimization matches. **The window does not**: the code takes 2 bins left, the
peak, and 2 bins right - five points, symmetric - where the document specifies
3 left and 2 right. ``mode="clip"`` keeps the window in range near the array
edges by repeating the edge bin.

Initial guess:

.. math::

   u^{(0)} = \sqrt{E_{\mathrm{peak}}} \cdot \sqrt{2e/m_p}\,/\,10^3
   \qquad
   n^{(0)} = 5 \left(\frac{400}{u^{(0)}}\right)^2
   \qquad
   T^{(0)} = 60000 \left(\frac{u^{(0)}}{400}\right)^2

i.e. the speed corresponding to the peak bin, and density/temperature scaled
from nominal 400 km/s values.

Failure handling
^^^^^^^^^^^^^^^^

**[CODE]** A fit is rejected if any of:

* ``curve_fit`` raises ``RuntimeError`` (no convergence);
* the returned covariance matrix is not finite (scipy may be echoing back the
  initial guess);
* :math:`R^2 < 0.7` on the fitted window.

On rejection, **the speed is still reported** - from the initial guess, i.e.
the peak-bin speed - and density and temperature are set to ``NaN``:

.. code-block:: python

   if sol is None:
       sol = initial_param_guess.copy()
       sol[1:] = np.nan

This mirrors the L3a behaviour the document describes for the science product
("whenever the fit fails and fill values are reported, ``PROTON_SW_SPEED`` is
still populated based on the peak of the sweep-averaged coincidence rates"),
though the :math:`R^2 \ge 0.7` threshold is the code's own - the document
specifies :math:`R^2 \ge 0.9` for the L3a fits and nothing for I-ALiRT.

Averaging
^^^^^^^^^

**[CODE]** ``geometric_mean(...)``. Sweeps with ``NaN`` in any of the three
parameters are excluded; if none of the 5 are valid, the record reports all
three as ``NaN`` with the arithmetic mean MET. Otherwise each parameter is
averaged in log space:

.. math::

   \bar{x} = \exp\left(\frac{1}{N}\sum_i \ln x_i\right)

and the record's ``swapi_epoch`` is the arithmetic mean of the valid sweeps'
midpoint METs, converted with ``met_to_ttj2000ns``.

The 5-sweep window only emits when the sweeps are contiguous:

.. code-block:: python

   if len(swapi_met_list) >= 5 and np.all(
       np.isclose(np.diff(swapi_met_list[-5:]), 12.0, atol=0.05)
   ):

so a data gap suppresses records until 5 clean sweeps have accumulated again.

Output record
-------------

**[CODE]** One dict per emitted window, values as ``Decimal`` to 3 decimal
places (or ``None`` when not finite), merged with the standard I-ALiRT
instrument header:

.. code-block:: python

   {
     ... instrument header items ...
     "instrument": "swapi",
     "swapi_epoch": <int, TT-J2000 ns>,
     "swapi_pseudo_proton_speed": Decimal | None,
     "swapi_pseudo_proton_density": Decimal | None,
     "swapi_pseudo_proton_temperature": Decimal | None,
   }

Reference numbers
-----------------

**[DOC]** Figure 24 is the validation case: a simulated coarse sweep for
:math:`u = 550` km/s, :math:`n = 5.27` cm\ :sup:`-3`,
:math:`T = 1 \times 10^5` K, generated with the **full** instrument response
function, then fit with the analytical model. Recovered:
:math:`n_p = 4.67 \pm 0.17` cm\ :sup:`-3`,
:math:`u_p = 545.3 \pm 1.24` km/s,
:math:`T_p = (1.18 \pm 0.08) \times 10^5` K.

That is the accuracy to expect: speed close, density and temperature "within
acceptable ranges". **[DOC]** "Future improvements may be made to correct the
values of pseudo speed, density, and temperature based on the pre-computed LUT
for a range of input parameters" - i.e. a bias-correction table, not yet
delivered.

Tests
-----

**[CODE]** ``imap_processing/tests/ialirt/unit/test_process_swapi.py``, which
exercises ``count_rate``, ``optimize_pseudo_parameters``,
``geometric_mean`` and the full ``process_swapi_ialirt`` including the returned
dictionary keys.
