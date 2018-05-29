""" Test forward propagation methods for FFNs.
"""

def test_fwdprop_number_adder():
    """test given unit wieghts will make a number adder.
    """
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset
    from crpm.fwdprop import fwdprop

    #manually create shallow bodyplan with 5 inputs and 1 output
    bodyplan = [{"layer":0, "n":5, "activation":"identity"}]
    bodyplan.append({"layer":1, "n":1, "activation":"linear"})

    #create model
    model = init_ffn(bodyplan)

    #manually set layer 1 weights to 1 and biases to 0
    w_1 = model[1]["weight"]
    model[1]["weight"] = w_1.fill(1.0)

    #run forward propagation with example data in number_adder.csv
    __, data = load_dataset("crpm/data/number_adder.csv")
    indepvars = data[0:4,]
    depvars = data[5,]
    prediction, state = fwdprop(indepvars, model)

    print(depvars)
    print(prediction)
    print(depvars == prediction)


    assert all(depvars == prediction)
