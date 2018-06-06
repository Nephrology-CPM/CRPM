#!/bin/bash

python -W ignore jdrf_clin_3met.py clin_3met_bodyplan.csv data/eGFR60MA.csv 100
python -W ignore jdrf_clin_3met.py clin_3met_bodyplan10.csv data/eGFR60MA.csv 100

python -W ignore jdrf_clin.py clin_bodyplan.csv data/eGFR60MA.csv 100
python -W ignore jdrf_clin.py clin_bodyplan10.csv data/eGFR60MA.csv 100

python -W ignore jdrf_clin_3met.py clin_3met_bodyplan.csv data/fullcohort.csv 100
python -W ignore jdrf_clin_3met.py clin_3met_bodyplan10.csv data/fullcohort.csv 100

python -W ignore jdrf_clin.py clin_bodyplan.csv data/fullcohort.csv 100
python -W ignore jdrf_clin.py clin_bodyplan10.csv data/fullcohort.csv 100
