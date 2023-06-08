# pywatershed-csdms-2023

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jmccreight/pywatershed-csdms-2023/main)



The enclosed notebooks are EPubs for the [May 2023 CSDMS conference](https://csdms.colorado.edu/wiki/Form:Annualmeeting2023). 
The goal of this repository is to demonstrate how to use the `pywatershed` package for hydrologic modeling and highlight some of its applications.

## Contents description

### `README.md`
This overview file.

### `LICENSE`
The license applied to this work.

### `env/`
Conda/mamba environment files that support the notebooks in this repository. Conda (or mamba) is required to install the environment. The `pws-env.yml` environment works as of May 2023. However it may become outdated and it can take a very long time to install using conda. For that reason there are `pws-env-frozen-<platform>-<arch>.yml` files which should install much more quickly in conda. These have specific instructions for conda installation at the top of the individual files, wher you choose the `env_name` you want for a given `env_file`:
```
$ conda remove -y --name <env_name> --all  # if it exists
$ conda create -y --name <env_name>
$ conda env update --name <env_name> --file <env_file>.yml
```
Generally, solving these environments was much quicker using mamba than conda, but doing so can be somewhat unclear as mamba is newer and still under development. 


### 01_process_models.ipynb: [github](https://github.com/jmccreight/pywatershed-csdms-2023/blob/main/01_process_models.ipynb) | [nbviewer](https://nbviewer.org/github/jmccreight/pywatershed-csdms-2023/blob/main/01_process_models.ipynb)  
This jupyter notebook with its accompanying directory `process_models/` demonstrates how to construct, run, and probe a version of the National Hydrologic Model and its submodels in pywatershed.

### 02_sagehen_pws_mf6api.ipynb: [github](https://github.com/jmccreight/pywatershed-csdms-2023/blob/main/02_sagehen_pws_mf6api.ipynb) | [nbviewer](https://nbviewer.org/github/jmccreight/pywatershed-csdms-2023/blob/main/02_sagehen_pws_mf6api.ipynb)   
This jupyter notebook with its accompanying directory `sagehen_pws_mf6api/` demonstrates how to couple pywatershed to MODFLOW 6 via its BMI interface, resulting in a NHM-driven surface water - groundwater flow model.


Disclaimer
==========

This information is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The information has not received final approval by the U.S. Geological Survey (USGS) and is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the information.

From: https://www2.usgs.gov/fsp/fsp_disclaimers.asp#5

This software is in the public domain because it contains materials that originally came from the U.S. Geological Survey, an agency of the United States Department of Interior. For more information, see the [official USGS copyright policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits "official USGS copyright policy")

Although this software program has been used by the USGS, no warranty, expressed or implied, is made by the USGS or the U.S. Government as to the accuracy and functioning of the program and related program material nor shall the fact of distribution constitute any such warranty, and no responsibility is assumed by the USGS in connection therewith.
This software is provided "AS IS."
