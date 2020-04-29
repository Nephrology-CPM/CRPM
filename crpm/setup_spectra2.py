""" set up models for spectra2 dataset
"""

def setup_spectra2():
    """ will return model and downloaded data."""

    import numpy as np
    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from  bodyplan file
    bodyplan = read_bodyplan("crpm/data/spectra2_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download data
    data = np.load("crpm/data/spectra2.npz")

    #get list of keys in data (represents individual arrays)
    keylist = []
    for key in data.keys():
        keylist.append(key)

    return model, data[keylist[0]]
