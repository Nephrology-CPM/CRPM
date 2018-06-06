# CRPM
Data tools for the Center for Renal Precision Medicine at the University of Texas Health San Antonio.


## Getting Started
+ Follow directions to download miniconda installer for [Windows](https://conda.io/docs/user-guide/install/windows.html), [MacOS](https://conda.io/docs/user-guide/install/macos.html), or [Linux](https://conda.io/docs/user-guide/install/linux.html).
Run the installer script, e.g. for MacOS open a terminal and run:
```
bash Miniconda3-latest-MacOSX-x86_64.sh
```
which will open an installer screen that will walk you through installation.
+ Double check conda is updated.
```
conda update -n base conda
```
+ Create a virtual environment named 'CRPM'
```
conda create --file requirements.txt -c conda-forge -n CRPM
```
When prompted new packages will be installed, procceed by pressing `y`.
+ Activate `CRPM` virtual environment
```
conda activate CRPM
```
+ Update virtual environment (when necessary)
```
conda env update -n CRPM -f requirements.txt
```
+ Integration testing (for devs)- Check for code style and run tests
```
./integration.sh
```
+ To deactivate environment, use
```
conda deactivate
```
+ To remove an environment, use
```
conda env remove -n CRPM
```

## Running JDRF analysis
Before you begin make sure you have the `CRPM` environment activated. Refer to
the `Gettting Started` section for instructions on how to do this.

To run model with clinical variables + 3 metabolites
```
python -W ignore jdrf_clin_3met.py clin_3met_bodyplan.csv data/<dataset>.csv <nsamples>
```
where `<nsamples>` is an integer representing the number of models to train for
calculating the confidence interval on the performance measures
and `<dataset>` is either "fullcohort" or "eGFR60MA".
Data is arranged with samples in columns and variables
in rows such that data should have subject index in first column,
followed by rapid_progressor binary labels in second column,
followed by 10 variables (order should not matter):
"age","sex","diabetes_duration",
"baseline_a1c","egfr_v0","acr_v0","systolic_bp_v0",
"u_x3_methyl_crotonyl_glycine_v0_gcms_badj",
"u_citric_acid_v0_gcms_badj",
"u_glycolic_acid_v0_gcms_badj".

To run model with clinical variables
```
python -W ignore jdrf_clin.py clin_bodyplan.csv data/<dataset>.csv <nsamples>
```
where `<nsamples>` is an integer representing the number of models to train for
calculating the confidence interval on the performance measures
`<dataset>` is either "fullcohort" or "eGFR60MA".
Data is arranged with samples in columns and variables
in rows such that data should have subject index in first column,
followed by rapid_progressor binary labels in second column,
followed by 7 variables (order should not matter):
"age","sex","diabetes_duration",
"baseline_a1c","egfr_v0","acr_v0","systolic_bp_v0".

Additionally, a batch submission script is available `batchjob.sh` that will
run all models **_(can take up to several hours to complete depending on your machine)_**

## Generating the Datasets
The R-markdown file [jdrf.Rmd](jdrf.Rmd) provided was used to generate the
datasets from the JDRF rawdata (not provided). In summary:
+ acr is log2 transformed.
+ metabolite data is log2 transformed then mean centered and transformed for
 unit variance.
+ sex is binary classifier is shifted from (1,2) to values 0 and 1


## Project Goals
+ implement a deep neural network framework **done**
  + forward and back propagation **done**
  + body plan **done**
  + activation functions **done**
+ implement a self organizing map framework
+ implement a Monte-Carlo hyper-parameterization routine for deep neural networks
