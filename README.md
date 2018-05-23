# CRPM
Data tools for the Center for Renal Precision Medicine at the University of Texas Health San Antonio.


## Getting Started
+ Follow directions to install miniconda for [Windows](https://conda.io/docs/user-guide/install/windows.html), [MacOS](https://conda.io/docs/user-guide/install/macos.html), or [Linux](https://conda.io/docs/user-guide/install/linux.html).
For example for MacOS open a terminal window and run:
```
bash Miniconda3-latest-MacOSX-x86_64.sh
```
which will open an installer screen that will walk you through installation.
+ Double check conda is updated.
```
conda update -n base conda
```
+ Set up `CRPM` virtual environment using provided `environment.yml` file
```
conda env create
```
+ Update virtual environement (when necessary)
```
conda env update
```
+ Activate `CRPM` virtual environment
```
source activate CRPM
```
+ Export environment to `requirements.txt` (for explicit builds)
```
conda list -e > requirements.txt
```
+ Integration testing - Check for code style and run tests
```
./integration.sh
```
