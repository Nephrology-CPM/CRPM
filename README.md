# CRPM
Data tools for the Center for Renal Precision Medicine at the University of Texas Health San Antonio.


# Getting Started
It is recomended you install in some kind of self contained environment so your
python settings don't get messed with. There are a few options such as
[virtualenv](http://pypi.org/project/virtualenv)
or [conda](https://docs.conda.io/projects/conda/en/latest/) which we will walk
you through now.

## (Option 1) Setup a virtual environment with virtualenv
Before you go any further, make sure you have Python and that it’s available
from your command line. You can check this by simply running:
```
python --version
```
You should get some output like 3.6.2. If not you may have python3 but have to call it by name.
```
python3 --version
```
If it turns out you do not have Python, please install
the latest 3.x version from [python.org](python.org).
If you installed Python from source, with an installer from
[python.org](python.org), or via [Homebrew](https://brew.sh/) you should already
have pip. If you’re on Linux and installed using your OS package manager, you may have to install pip separately.

[venv](https://docs.python.org/3/library/venv.html) is a tool to create isolated
Python environments. venv creates a folder which contains all the
necessary executables to use the packages that a Python project would need.

1. Create a virtual environment named 'venv' for this project in this
project's directory:
```
cd path/to/this/project
python3 -m venv venv
```
2. Activate the virtual environment to begin using it:
```
source venv/bin/activate
```
The name of the current virtual environment will now appear on the left of the
prompt (e.g. (venv)Your-Computer:project_folder UserName$) to let you know that
it’s active.

For Windows, the same command mentioned in step 1 can be used to create a
virtual environment. However, activating the environment requires a slightly
different command. Assuming that you are in your project directory the command is.
```
C:\Users\SomeUser\project_folder> venv\Scripts\activate
```
3. To deactivate the virtual environment when you are done using it use the command
```
deactivate
```


## (Option 2) Setup a virtual environment with conda
The conda package and environment manager is included in all versions of
[Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary),
[Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary),
and [Anaconda Repository](https://docs.continuum.io/anaconda-repository/).
Here we will go with the lightweight miniconda option.
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

# Install with Makefile
A Makfile is provided to make installation easy. If you deactivated your virtual
environment go ahead and activate it again. Make sure the virtual environment is
active and install with the command
```
make
```
Devs can test the code with the command
```
make test
```

## Project Goals
+ implement a self organizing map framework (done)
+ implement a Monte-Carlo hyper-parameterization routine for deep neural networks
+ implement a new stochastic learing algorithm (done)
