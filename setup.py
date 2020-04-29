from setuptools import setup, find_packages

setup(
    name='crpm',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    install_requires=['numpy', 'scipy', 'progressbar'],
    include_package_data=True,
    package_data={
        '': ['*.csv'],
    },
    license='MIT',
    description='Data tools for the Center for Renal Precision Medicine at the University of Texas Health San Antonio.',
    long_description=open('README.md').read(),
    #install_requires=[''],
    url='https://github.com/dmontemayor/CRPM',
    author='Daniel Montemayor',
    author_email='montemayord2@uthscsa.edu',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
)
