""" set up linear regression model for multicorrel dataset
"""

def setup_multicorrel():
    """ will return model and downloaded data."""

    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #manually create a bodyplan for multicorrel.csv data
    bodyplan = [
        {"layer":0, "n":3, "activation":"linear"},
        {"layer":1, "n":1, "activation":"linear", "regval":0}
        ]

    #create model
    model = init_ffn(bodyplan)

    #download nestedCs data
    __, data = load_dataset("crpm/data/multicorrel.csv")

    return model, data
