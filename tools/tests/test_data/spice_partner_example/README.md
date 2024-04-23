# Instructions for developers

## Required SPICE kernels

The following SPICE kernels are required to run the program:

1. IMAP_launch20250429_1D.bsp
2. imap_wkcp.tf
3. pck00010.tpc
4. imap_sclk_0000.tsc
5. naif0012.tls
6. de440.bsp

The first is an IMAP-specific ephemeris kernel.
The second is an IMAP-specific frame kernel.
All others are generic NAIF SPICE kernels.

## SPICE tools

A description of all spice libraries used in the program can be found at the following link:
https://spiceypy.readthedocs.io/en/main/documentation.html

## Description of results

The 'spice_output_data.txt' file contains the output results.
The IMAP spacecraft velocity (km/s) and position (km) are with respect to Earth in the J2000 reference frame.

time_et: ephemeral time
time_utc: time in UTC (human-readable)
x_position: x-coordinate of the spacecraft (km)
y_position: y-coordinate of the spacecraft (km)
z_position: z-coordinate of the spacecraft (km)
x_velocity: x-component of the spacecraft velocity (km/s)
y_velocity: y-component of the spacecraft velocity (km/s)
z_velocity: z-component of the spacecraft velocity (km/s)

## Important information

The time interval between measurements was selected to be 1e4 seconds.
That time interval can be changed so that there is a small cadence between measurements.
