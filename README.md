# BC-IRF

### Create a pulse run

Prep_pulses notebook: importing the file of interest to pulse, selecting the region of interest for the pulse. Add a reasonable amount relative to the existing emissions amounts (we use +1x and +15x). When running GEOS-Chem modify the normal file to read in the new one instead via HEMCO_Config.

### Modify GCHP stretched data to be useable in a lat lon format

regrid_gchp_stretched.py: regrid the stretched data
regrid_gchp_stretched.ipynb: original notebook for this/for testing
batch_regrid_stretched.py: submits a large number of regriddings

### Shutdown scenarios and convolution

prep_population_gridded_data.py: preps the population data so we can calculate health impacts in shutdowns_GAINS.py
shutdowns_GAINS.py: shutdowns and convolves plants, select for one of three types of shutdown approaches
batch_shutdown.py: submits a large number of shutdowns
shutdowns_GAINS_test.ipynb: test for shutdowns_GAINS.py

### Create Emissions Factors for the Plants
BC_emis_factors.ipynb: combines data from GAINS and Han Springer et al work to create emissions factors of BC for each plant


### Scenario plotting
contours_plotting.ipynb: plot the contours that are output from shutdowns_GAINS.py
shutdowns_emis_plots.ipynb: plot the scenarios and emissions and contributions
video_countries.ipynb: create videos of country level pollution

### Green's Functions
GF_mean_times.py: create the GF
GF_mean_times.ipynb: original notebook to create GF/for testing
GF_mean_times-plotting.ipynb: plot our GF for comparisons
GF_pulse_vs_step.ipynb: methods testing of a step vs. pulse-- important for comparison to source receptor


### Utilities
utils.py: utilities/functions
