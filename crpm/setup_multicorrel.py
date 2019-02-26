""" set up linear regression model for multicorrel dataset
"""

def setup_multicorrel():
    """ will return model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from  bodyplan file
    bodyplan = read_bodyplan("crpm/data/multicorrel_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download nestedCs data
    __, data = load_dataset("crpm/data/multicorrel.csv")

    return model, data


def setup_multicorrel_c():
    """ will return model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from  bodyplan file
    bodyplan = read_bodyplan("crpm/data/multicorrel_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download nestedCs data
    __, data = load_dataset("crpm/data/multicorrel_C.csv")

    return model, data

def setup_multicorrel_deep_c():
    """ will return deep model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from  bodyplan file
    bodyplan = read_bodyplan("crpm/data/multicorrel_deep_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download nestedCs data
    __, data = load_dataset("crpm/data/multicorrel_C.csv")

    return model, data
