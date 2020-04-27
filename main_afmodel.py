""" Generate 2 group spectra of K modes """
import numpy as np
import matplotlib.pyplot as plt
from crpm.af_model import AFnetEnvironment

def afmodel():
    """ Return data set of 600 patients in two cohorts.
    """
    #get cohort1
    cohort1 =AFnetEnvironment(patients=300, typeratio=0.6)
    #drop X, Y, and time feature in state found on last 3 rows
    cohort1.state = cohort1.state[0:-3, :]

    #get cohort2
    cohort2 =AFnetEnvironment(patients=300, typeratio=0.4)
    #drop X, Y, and time feature in state found on last 3 rows
    cohort2.state = cohort2.state[0:-3, :]
    #degrade metabo B by 80%
    #cohort2.state[1, :] *= .8
    #degrade metabo D by 60%
    #cohort2.state[3, :] *= .6

    # save data
    np.savez("crpm/data/afmodel",
             cohort1.state,
             cohort1.group,
             cohort2.state,
             cohort2.group)

    #---Diagnostic---

    #plot metabolite distributions
    plt.violinplot(cohort1.state.T)
    plt.violinplot(cohort2.state.T)
    plt.show()

    #calculate metabo correlation matricies
    corr1 = np.corrcoef(cohort1.state)
    corr2 = np.corrcoef(cohort2.state)

    #visulaize correlations
    plt.matshow(corr1)
    plt.show()
    plt.matshow(corr2)
    plt.show()

    #zscore metabos may be bad because dist is saddle shaped
    data = np.hstack((cohort1.state, cohort2.state))
    data = np.divide(data-np.mean(data, axis=1, keepdims=True), np.std(data,
                     axis=1, keepdims=True))
    #plot metabolite distributions
    plt.violinplot(data.T)
    plt.show()



if __name__ == "__main__":
    afmodel()
