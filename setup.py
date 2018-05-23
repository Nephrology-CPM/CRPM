from setuptools import setup, find_packages

setup(
    name='CRPM',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Data tools for the Center for Renal Precision Medicine at the University of Texas Health San Antonio.',
    long_description=open('README.md').read(),
    install_requires=[''],
    url='https://github.com/dmontemayor/CRPM',
    author='Daniel Montemayor',
    author_email='montemayord2@uthscsa.edu'
)
