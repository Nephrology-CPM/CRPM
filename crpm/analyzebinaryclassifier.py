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
            sensitivity: a real number between 0 and 1 representing the
            minimum sensitivity desired.
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

    #true data stats
    totalpop = len(targets)
    condpos = (targets == 1)
    condneg = (targets == 0)
    sumpos = np.sum(condpos)
    sumneg = np.sum(condneg)
    prevalence = sumpos/totalpop

    #init ROC
    roc = []

    #init max accuracy
    max_acc = 0

    #define ROC resolution
    npt = 1000
    resolution = 1.0/npt

    #calculate ROC 1000 point scale (.001 resolution)
    for i in range(npt):

        #calculate threshold
        threshold = i * resolution

        #get predicted classes
        predpos = (pred > threshold) #positive predictions
        predneg = ~predpos #negative predictions

        truepos = np.sum(predpos & condpos)
        trueneg = np.sum(predneg & condneg)
        falsepos = np.sum(predpos & condneg)

        #calculate false positive rate
        fpr = falsepos/sumneg

        #calculate sensitivity
        tpr = truepos/sumpos

        #calculate accuracy
        acc = (truepos + trueneg)/totalpop #accuracy

        #record best Accuracy
        if acc >= max_acc:
            max_acc = acc
            optimal_threshold = threshold

        #push to ROC
        roc.append([fpr, tpr])

    # get stats at optimal_threshold
    predpos = (pred > optimal_threshold) #positive predictions
    predneg = ~predpos #negative predictions
    sumpredpos = np.sum(predpos)
    sumpredneg = np.sum(predneg)

    truepos = np.sum(predpos & condpos)
    trueneg = np.sum(predneg & condneg)
    falsepos = np.sum(predpos & condneg)
    falseneg = np.sum(predneg & condpos)

    acc = (truepos + trueneg)/totalpop #accuracy
    tpr = truepos/sumpos #sensitivity
    tnr = trueneg/sumneg #specificity
    fnr = falseneg/sumpos #false negative rate
    fpr = falsepos/sumneg #false positive rate
    ppv = truepos/sumpredpos #positive predicitive value
    npv = trueneg/sumpredneg #negative predictive value
    fdr = falsepos/sumpredpos #false discovery rate
    fomr = falseneg/sumpredneg #false omission rate
    lrpos = tpr/fpr #positive likelihood ratio
    lrneg = fnr/tnr #negative likelihood ratio
    dor = lrpos/lrneg #diagnostic odds ratio
    f1score = 2/(1/tpr+1/ppv)#F1 score

    report = {"AreaUnderCurve":np.sum(roc)/npt,
              "OptimalThreshold":optimal_threshold,
              "Accuracy":acc, "Prevalence":prevalence,
              "Sensitivity":tpr, "Specificity":tnr,
              "FalseNegRate":fnr, "FalsePosRate":fpr,
              "PosPredValue":ppv, "NegPredValue":npv,
              "FalseDiscRate":fdr, "FalseOmitRate":fomr,
              "PosLikelihoodRatio":lrpos, "NegLikelihoodRatio":lrneg,
              "DiagnosticOddsRatio":dor, "F1Score":f1score,

             }

    return roc, report
