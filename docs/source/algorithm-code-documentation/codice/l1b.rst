.. _codice-l1b:

Level 1B - Counts to Rates
==========================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**[DOC]** Section 11. **[CODE]** ``codice_l1b.py`` - 219 lines, one function
that does everything.

L1B is conceptually the simplest level in CoDICE: divide counts by an
accumulation time. All of the difficulty is in knowing **what that accumulation
time is for each product**, which depends on how many spin sectors and spins the
on-board collapse summed together.

The general form
----------------

**[DOC]**

.. math::

   R_j = \frac{C_j}{\tau_{FSW}}, \qquad
   \tau_{FSW}(n_{sectors}, n_{spins}) = t_{acquire} \cdot n_{sectors} \cdot n_{spins}

where :math:`t_{acquire}` is the time spent acquiring data in a single 15 deg
spin sector (for Hi) or in a single (ESA step, sector) pair (for Lo). See
:ref:`codice-esa-stepping` for the Appendix C derivation.

Uncertainties are converted the same way: :math:`\sigma_R = \sigma_C /
\tau_{FSW}`.

**[CODE]** ``convert_to_rates(dataset, descriptor, cdf_attrs)`` picks the
variable list by name reflection::

    variables_to_convert = getattr(
        constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
    )

so ``lo-sw-species`` -> ``LO_SW_SPECIES_VARIABLE_NAMES``. **A descriptor with no
matching constant raises ``AttributeError``** - which is how the missing
products fail.

Uncertainties are skipped entirely when ``"counters" in descriptor``
(``calculate_unc = False``), matching the document's treatment of the
engineering counters as diagnostics.

CoDICE-Hi
---------

**[DOC]** Section 11.1. :math:`t_{acquire} = 0.59916` s, configurable via LUT.

.. list-table::
   :header-rows: 1
   :widths: 30 16 16 38

   * - Product
     - :math:`n_{sectors}`
     - :math:`n_{spins}`
     - Document reference
   * - Instrument rates (both)
     - 24
     - 16
     - 11.1.1
   * - Omni-directional species rates
     - 24
     - 4
     - 11.1.2
   * - Sectored species rates
     - 2
     - 16
     - 11.1.3
   * - Priority rates
     - 24
     - 16
     - 11.1.4
   * - I-ALiRT
     - 6
     - 4
     - 14.1

**[CODE]** ``L1B_DATA_PRODUCT_CONFIGURATIONS`` in ``constants.py`` holds exactly
these values, and the Hi branch is:

.. code-block:: python

   denominator = (
       constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spin_sectors"]
       * constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spins"]
       * constants.HI_ACQUISITION_TIME
   )

so the denominator is a **scalar** for all Hi products.

CoDICE-Lo
---------

**[DOC]** Section 11.2. Lo is different because :math:`t_{acquire}` **varies
per ESA step** - it depends on how many ESA steps were sampled in the half-spin
that step belonged to. The document writes

.. math::

   t_{accum} = t_{acquire} \times 10^{-3} \times n_{sectors}

with :math:`t_{acquire}` in milliseconds.

**[CODE]** ``acquisition_time_per_esa_step`` is written at L1A **already in
seconds** as an ``(epoch, esa_step)`` array (``calculate_acq_time_per_step``
divides by 1e3 before returning), so L1B multiplies by the sector count only.
The denominator is therefore a 2-D array that broadcasts against the species
data.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Product
     - :math:`n_{sectors}`
     - Document reference
   * - Instrument rates (aggregated, singles)
     - 2
     - 11.2.1 - two 15 deg sectors summed into a 30 deg bin
   * - Priority rates (SW and NSW)
     - 1
     - 11.2.4
   * - Species rates
     - **12, except 11 at ESA step 127**
     - 11.2.2
   * - Angular rates
     - 1
     - 11.2.3 - **not implemented**
   * - I-ALiRT
     - 12 / 11
     - 14.2.1

The flyback step
^^^^^^^^^^^^^^^^

**[DOC]** "Due to the ESA stepping scheme, the last spin sector at ESA step 127
is used as a flyback step and so counts are not accumulated." Hence
:math:`n_{sectors} = 12` for steps 0-126 and **11 for step 127**.

**[CODE]**

.. code-block:: python

   n_sector = xr.full_like(dataset.acquisition_time_per_esa_step, 12.0,
                           dtype=np.float64)
   n_sector[:, -1] = 11.0
   denominator = dataset.acquisition_time_per_esa_step * n_sector

Applied to ``lo-sw-species`` and ``lo-ialirt`` only.

.. note::

   Under the post-2025-12-18 stepping scheme the sweep ends at **step 103**, not
   127, so index ``-1`` is a padded, NaN-masked step and the 11-sector
   correction lands on a step that carries no data. That is harmless (NaN either
   way) but it means the flyback correction is effectively inactive for P2/P3
   data. Confirm with the CoDICE team whether step 103 now needs the correction.

Energy per charge
-----------------

**[DOC]** For every Lo product:

.. math::

   \mathrm{energy\_table}(l) = \mathrm{voltage\_table}(l) \times k \times 10^{-3}
   \quad [\mathrm{keV/e}]

**[CODE]** Computed once for any descriptor starting with ``lo-``:

.. code-block:: python

   energy_per_charge = (dataset["voltage_table"].values
                        * dataset["k_factor"].values * 1e-3)

and written as both ``energy_per_charge`` and a string
``energy_per_charge_label`` (``f"{value:.3f}"``). The document calls the variable
``energy_table``; the code calls it ``energy_per_charge``.

``cdf_attrs`` is optional here specifically so that the I-ALiRT pipeline can
call ``convert_to_rates`` for its intermediate values without a CDF attribute
manager.

Variables dropped at L1B
------------------------

**[CODE]** Metadata that only mattered for unpacking is removed:

* **Lo counters and priority**: ``k_factor``, ``nso_half_spin``,
  ``sw_bias_gain_mode``, ``st_bias_gain_mode``, ``spin_period``,
  ``voltage_table``, ``nso_esa_step``, ``nso_spin_sector``.
* **Lo species and I-ALiRT**: the same minus ``nso_esa_step`` and
  ``nso_spin_sector`` (these two are **kept**, because L2 needs them).
* **All products**: ``spin_period`` if still present.

``acquisition_time_per_esa_step`` is deliberately **not** dropped, with the
comment ``# TODO: undo this when I get new validation file from Joey``. It is
dropped later, at L2.

Note that ``rgfo_half_spin``, ``rgfo_esa_step``, ``rgfo_spin_sector``,
``half_spin_per_esa_step`` and ``packet_version`` survive L1B for the Lo species
product - L2 needs every one of them to pick the right geometric factor.

Housekeeping
------------

**[DOC]** Section 11.3: "all voltages and currents are converted to physical
units".

**[CODE]** This is done by re-running ``packet_file_to_datasets`` with
``use_derived_value=True`` **inside ``process_l1a``**, so the XTCE polynomial
calibrators do the conversion. ``process_codice_l1b`` is never called for
``hskp`` - and could not be, since there is no ``HSKP_VARIABLE_NAMES``.

Entry point
-----------

.. code-block:: python

   def process_codice_l1b(file_path: Path) -> xr.Dataset:
       l1a_dataset = load_cdf(file_path)
       dataset_name = l1a_dataset.attrs["Logical_source"].replace("_l1a_", "_l1b_")
       descriptor = dataset_name.removeprefix("imap_codice_l1b_")
       ...
       l1b_dataset = l1a_dataset.copy(deep=True)
       l1b_dataset.attrs = cdf_attrs.get_global_attributes(dataset_name)
       return convert_to_rates(l1b_dataset, descriptor, cdf_attrs)

The descriptor is derived from the **input file's** ``Logical_source``, not from
the CLI ``--descriptor`` argument. L1B is a pure L1A-CDF-in, L1B-CDF-out
transformation with no ancillary inputs.
