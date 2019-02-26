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

## Project Goals
+ implement a self organizing map framework
+ implement a Monte-Carlo hyper-parameterization routine for deep neural networks
+ implement a new stochastic learing algorithm (done)
