""" set up models for afmodel dataset
"""

def setup_afmodel():
    """ will return model prototype and downloaded data."""

    import numpy as np
    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from  bodyplan file
    bodyplan = read_bodyplan("crpm/data/afmodel_bodyplan.csv")

    #create model
    prototype = init_ffn(bodyplan)

    #download data
    data = np.load("crpm/data/afmodel.npz")

    #get list of keys in data (represents individual arrays)
    keylist = []
    for key in data.keys():
        keylist.append(key)

    #return encoder protype, cohort1 data, cohort1 labels, cohort2 data, cohort2 labels 
    return prototype, data[keylist[0]], data[keylist[1]], data[keylist[2]], data[keylist[3]]
