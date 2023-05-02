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
