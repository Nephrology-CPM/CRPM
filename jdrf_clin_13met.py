""" JDRF analysis by deep NN incorporating metabolites with clinical vars.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from crpm.ffn_bodyplan import read_bodyplan
from crpm.ffn_bodyplan import init_ffn
from crpm.dataset import load_dataset
from crpm.gradientdecent import gradientdecent
from crpm.analyzebinaryclassifier import analyzebinaryclassifier

def jdrf_clin_13met(bodyplanfile,datafile,nsample=100):
    """ compare JDRF models with clinical variables and 3 metabolites
    """

    #read bodyplan
    bodyplan = read_bodyplan(bodyplanfile)

    #manually create a bodyplan for JDRF data
    #bodyplan = [
    #    {"layer":0, "n":10, "activation":"linear"},
    #    {"layer":1, "n":1, "activation":"logistic"}
    #    ]

    #download jdrf data
    #__, data = load_dataset("jdrf_data/eGFR60MA.csv")
    #__, data = load_dataset("jdrf_data/fullcohort.csv")

    #data should have subject index in first column,
    #followed by rapid_progressor binary labels in second column,
    #followed by 10 variables (order should not matter):
    #"age","sex","diabetes_duration",
    #"baseline_a1c","egfr_v0","acr_v0","systolic_bp_v0",
    #"u_x3_methyl_crotonyl_glycine_v0_gcms_badj",
    #"u_citric_acid_v0_gcms_badj",
    #"u_glycolic_acid_v0_gcms_badj"
    __, data = load_dataset(datafile)

    # get training data
    train = data[2:,]
    target = data[1,]

    #init validation parameters
    CVreports = []
    nsets = 5 #definition of K
    nroc = 5 #diagnostic param: number of ROCs to save
    #nsample = 1000 #number of validation sets to run

    #repeat K-fold CV untill we have nsamples
    while len(CVreports)<int(nsample):
        invalid = np.random.randint(0, nsets, data[1,].shape)
        i=0
        #while i < nsets: #for proper k-fold CV **comment out for simple 80-20 training
        #init model
        model = init_ffn(bodyplan)

        #train model
        pred, __ = gradientdecent(model,
                                train[:,invalid != i],
                                target[invalid != i],
                                "mse",
                                train[:,invalid == i],
                                target[invalid == i])
        #analyze binary classifier
        roc, report = analyzebinaryclassifier(pred, target[invalid == i])

        #accept only error free reports
        hasnan = any(np.isnan(val) for val in report.values())
        hasinf = any(np.isinf(val) for val in report.values())
        if  not (hasnan or hasinf):
            #save reports
            CVreports.append({"report":report})
            #i += 1
            print(len(CVreports),flush=True)
            #-- diagnostic -- plot roc for this k set
            #if len(CVreports) < nroc:
                #plt.scatter(*zip(*roc))

    #collate report statistics
    reportstats = {}
    for report in CVreports:
        for key, val in report["report"].items():
            reportstats.setdefault(key, []).append(val)
    #print out report statistics
    for key, val in reportstats.items():
        ave = np.mean(val)
        print(key, ave, st.t.interval(0.95, len(val)-1, loc=ave, scale=st.sem(val)),flush=True)


    ##-- diagnostic -- show roc plots. warning: only do for a small number of ROCS
    #plt.show()


if __name__ == '__main__':
    jdrf_clin_13met(sys.argv[1], sys.argv[2], sys.argv[3])
