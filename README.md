# pywatershed-csdms-2023

The enclosed notebooks are EPubs for the May 2023 CSDMS conference. 
The goal of this repository is to demonstrate how to use the `pywatershed` package and highlight some of its applications.

## Contents description

### `README.md`
This overview file.

### `LICENSE`
The license applied to this work.

### `pws-env.yml`
The conda environment file that supports all notebooks in this repository. Conda is required to install as follows 
```
conda env create -f environment.yml
```
which is described [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) in more detail.

### 01_process_models.ipynb                                  
This jupyter notebook with its accompanying directory `process_models/` demonstrates how to construct, run, and probe a version of the National Hydrologic Model and its submodels in pywatershed.

### 02_sagehen_pws_mf6api.ipynb
This jupyter notebook with its accompanying directory `sagehen_pws_mf6api/` demonstrates how to couple pywatershed to MODFLOW 6 via its BMI interface, resulting in a NHM-driven surface water - groundwater flow model.


Disclaimer
==========

This information is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The information has not received final approval by the U.S. Geological Survey (USGS) and is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the information.

From: https://www2.usgs.gov/fsp/fsp_disclaimers.asp#5

This software is in the public domain because it contains materials that originally came from the U.S. Geological Survey, an agency of the United States Department of Interior. For more information, see the [official USGS copyright policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits "official USGS copyright policy")

Although this software program has been used by the USGS, no warranty, expressed or implied, is made by the USGS or the U.S. Government as to the accuracy and functioning of the program and related program material nor shall the fact of distribution constitute any such warranty, and no responsibility is assumed by the USGS in connection therewith.
This software is provided "AS IS."
