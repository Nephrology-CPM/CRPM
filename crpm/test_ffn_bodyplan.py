"""Test feed forward network body plan functions.
"""

def test_example_bodyplan():
    """Check example_ffn_bodyplan.csv is read as we expect
    """

    from crpm.ffn_bodyplan import read_bodyplan
    bodyplan = read_bodyplan("crpm/data/example_ffn_bodyplan.csv")

    #confirm 4 layers deep
    assert len(bodyplan) == 5

    #confirm parameters for each layer
    assert bodyplan[0]["layer"] == 0
    assert bodyplan[0]["n"] == 2
    assert bodyplan[0]["activation"] == "linear"

    assert bodyplan[1]["layer"] == 1
    assert bodyplan[1]["n"] == 3
    assert bodyplan[1]["activation"] == "relu"

    assert bodyplan[2]["layer"] == 2
    assert bodyplan[2]["n"] == 5
    assert bodyplan[2]["activation"] == "relu"

    assert bodyplan[3]["layer"] == 3
    assert bodyplan[3]["n"] == 7
    assert bodyplan[3]["activation"] == "relu"

    assert bodyplan[4]["layer"] == 4
    assert bodyplan[4]["n"] == 1
    assert bodyplan[4]["activation"] == "logistic"


def test_init_ffn():
    """Test ffn is created properly from example_bodyplan.csv
    """

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn

    bodyplan = read_bodyplan("crpm/data/example_ffn_bodyplan.csv")
    model = init_ffn(bodyplan)

    assert model[0]["layer"] == 0
    assert model[1]["layer"] == 1
    assert model[2]["layer"] == 2
    assert model[3]["layer"] == 3
    assert model[4]["layer"] == 4

    assert model[0]["n"] == 2
    assert model[1]["n"] == 3
    assert model[2]["n"] == 5
    assert model[3]["n"] == 7
    assert model[4]["n"] == 1

    assert model[1]["weight"].shape == (3, 2)
    assert model[2]["weight"].shape == (5, 3)
    assert model[3]["weight"].shape == (7, 5)
    assert model[4]["weight"].shape == (1, 7)

    assert model[1]["bias"].shape == (3, 1)
    assert model[2]["bias"].shape == (5, 1)
    assert model[3]["bias"].shape == (7, 1)
    assert model[4]["bias"].shape == (1, 1)

    assert model[0]["activation"] == 'linear'
    assert model[1]["activation"] == 'relu'
    assert model[2]["activation"] == 'relu'
    assert model[3]["activation"] == 'relu'
    assert model[4]["activation"] == 'logistic'


def test_init_ffn_types():
    """check if elements in layer dictionaries are of the correct type
    weights and biases should be ndarrays
    """
    import numpy as np

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn

    bodyplan = read_bodyplan("crpm/data/example_ffn_bodyplan.csv")
    model = init_ffn(bodyplan)

    for layer in model:
        assert isinstance(layer["layer"], int)
        assert isinstance(layer["n"], int)
        assert isinstance(layer["activation"], str)
        if layer["layer"] > 0:
            assert isinstance(layer["regval"], float)
            assert isinstance(layer["weight"], np.ndarray)
            assert isinstance(layer["bias"], np.ndarray)
