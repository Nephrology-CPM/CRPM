"""standard analysis for binary classifier models
"""

def analyzebinaryclassifier(pred, targets):
    """standard analysis for binary classifiers
        Args:
            pred: An array of shape (1,M) representing the predicted values
                where M is the total population.
                Predicted values range from 0 to 1.
            target: An array of shape (1,M) representing the true values of
                the binary classes with values of 0 or 1.
        Returns: The ROC and a dictionary of analysis results at the optimal
            threshold with key:value pairs
            "Prevalence":float,"Accuracy":float,
            "Sensitivity":float, "Specificity":float,
            "FalseNegRate":float, "FalsePosRate":float,
            "PosPredValue":float, "NegPredValue":float,
            "FalseDiscRate":float, "FalseOmitRate":float,
            "PosLikelihoodRatio":float, "NegLikelihoodRatio":float,
            "DiagnosticOddsRatio":float, "F1Score":float,
            "OptimalThreshold":float, "AreaUnderCurve":float.
            The optimal threshold is the threashold that maximizes the accuracy.
    """
    import numpy as np

    from crpm.trapintegration import trapintegration

    #constant data parameters
    sumpos = np.sum(targets == 1)
    sumneg = np.sum(targets == 0)

    #init ROC
    roc = []

    #default values
    var = {
        "alpha": np.nan,
        "acc":np.nan,
        "tpr":np.nan,
        "tnr":np.nan,
        "fnr":np.nan,
        "fpr":np.nan,
        "ppv":np.nan,
        "npv":np.nan,
        "fdr":np.nan,
        "fomr":np.nan,
        "lrpos":np.nan,
        "lrneg":np.nan,
        "dor":np.nan,
        "f1score":np.nan,
        }

    #check for valid pred data
    if not any(np.isnan(pred[0])):

        #define index order: arrange samples in order of values
        idx = np.argsort(pred)[0]

        #target conditions in index order
        condpos = (targets[idx] == 1)
        condneg = (targets[idx] == 0)

        #init max f1score
        max_f1 = 0

        for i in idx:

            #calculate threshold
            threshold = pred[0, i]

            #get predicted classes
            predpos = (pred[0, idx] >= threshold) #positive predictions

            truepos = np.sum(predpos & condpos)
            falsepos = np.sum(predpos & condneg)

            #calculate false positive rate
            var["fpr"] = falsepos/sumneg

            #calculate sensitivity
            var["tpr"] = truepos/sumpos

            #calculate positive predicitve value
            var["ppv"] = truepos/(truepos + falsepos)

            #calculate f1score
            var["f1score"] = 2/(1/var["tpr"]+1/var["ppv"])

            #record best F1score
            if var["f1score"] >= max_f1:
                max_f1 = np.copy(var["f1score"])
                var["alpha"] = np.copy(threshold)

            #push to ROC
            roc.append([var["fpr"], var["tpr"]])

        # get stats at threashold that optimizes f1score
        predpos = (pred[0, idx] >= var["alpha"]) #positive predictions

        truepos = np.sum(predpos & condpos)
        trueneg = np.sum(~predpos & condneg)
        falsepos = np.sum(predpos & condneg)

        var["acc"] = (truepos + trueneg)/len(targets)

        var["tpr"] = truepos/sumpos
        var["fnr"] = 1 - var["tpr"]

        var["fpr"] = falsepos/sumneg
        var["tnr"] = 1- var["fpr"]

        if np.sum(predpos) == 0:
            var["ppv"] = 1
            var["fdr"] = 0
        else:
            var["ppv"] = truepos/np.sum(predpos)
            var["fdr"] = 1 - var["ppv"]

        if np.sum(~predpos) == 0:
            var["npv"] = 1
            var["fomr"] = 0
        else:
            var["npv"] = trueneg/np.sum(~predpos)
            var["fomr"] = 1 - var["npv"]

        var["lrpos"] = var["tpr"]/var["fpr"]
        var["lrneg"] = var["fnr"]/var["tnr"]
        var["dor"] = var["lrpos"]/var["lrneg"]
        var["f1score"] = 2/(1/var["tpr"]+1/var["ppv"])

    report = {"AreaUnderCurve":trapintegration(roc),
              "OptimalThreshold":var["alpha"],
              "Accuracy":var["acc"], "Prevalence":sumpos/len(targets),
              "Sensitivity":var["tpr"], "Specificity":var["tnr"],
              "FalseNegRate":var["fnr"], "FalsePosRate":var["fpr"],
              "PosPredValue":var["ppv"], "NegPredValue":var["npv"],
              "FalseDiscRate":var["fdr"], "FalseOmitRate":var["fomr"],
              "PosLikelihoodRatio":var["lrpos"], "NegLikelihoodRatio":var["lrneg"],
              "DiagnosticOddsRatio":var["dor"], "F1Score":var["f1score"],
              "NPred":np.sum(predpos),
             }

    #print(report)
    return roc, report


def plotroc(roc):
    """ utility for visulaizing roc returned by analyzebinaryclassifier
    """
    import matplotlib.pyplot as plt
    plt.scatter(*zip(*roc))
    plt.show()
