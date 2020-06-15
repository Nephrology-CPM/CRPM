"""Test feed forward network class.
"""

def example_bodyplan_assertions(model):
    """assertions for example bodyplan"""

    #confirm 4 layers deep
    assert len(model.bodyplan) == 5
    assert len(model.body) == 5

    #confirm layers are numbered correctly
    assert model.body[0]["layer"] == 0
    assert model.body[1]["layer"] == 1
    assert model.body[2]["layer"] == 2
    assert model.body[3]["layer"] == 3
    assert model.body[4]["layer"] == 4

    #confirm correct number of nodes per layer
    assert model.body[0]["n"] == 2
    assert model.body[1]["n"] == 3
    assert model.body[2]["n"] == 5
    assert model.body[3]["n"] == 7
    assert model.body[4]["n"] == 1

    #check activation functions
    assert model.body[0]["activation"] == 'linear'
    assert model.body[1]["activation"] == 'relu'
    assert model.body[2]["activation"] == 'relu'
    assert model.body[3]["activation"] == 'relu'
    assert model.body[4]["activation"] == 'logistic'

    #---- following only apply to hidden layers ---

    #check shape of weight matricies
    assert model.body[1]["weight"].shape == (3, 2)
    assert model.body[2]["weight"].shape == (5, 3)
    assert model.body[3]["weight"].shape == (7, 5)
    assert model.body[4]["weight"].shape == (1, 7)

    #check shape of biases
    assert model.body[1]["bias"].shape == (3, 1)
    assert model.body[2]["bias"].shape == (5, 1)
    assert model.body[3]["bias"].shape == (7, 1)
    assert model.body[4]["bias"].shape == (1, 1)

    #check regularization type
    assert model.body[1]["lreg"] == 1
    assert model.body[2]["lreg"] == 1
    assert model.body[3]["lreg"] == 1
    assert model.body[4]["lreg"] == 1

    #check regularization term
    assert model.body[1]["regval"] == 0
    assert model.body[2]["regval"] == 0
    assert model.body[3]["regval"] == 0
    assert model.body[4]["regval"] == 0

    #check shape of weight momentum matricies
    assert model.body[1]["weightdot"].shape == (3, 2)
    assert model.body[2]["weightdot"].shape == (5, 3)
    assert model.body[3]["weightdot"].shape == (7, 5)
    assert model.body[4]["weightdot"].shape == (1, 7)

    #check shape of bias momenta
    assert model.body[1]["biasdot"].shape == (3, 1)
    assert model.body[2]["biasdot"].shape == (5, 1)
    assert model.body[3]["biasdot"].shape == (7, 1)
    assert model.body[4]["biasdot"].shape == (1, 1)


def test_init_ffn_from_file():
    """Test ffn is created properly from example_bodyplan.csv
    """

    from crpm.ffn import FFN

    #create FFN from file
    model = FFN("crpm/data/example_ffn_bodyplan.csv")

    example_bodyplan_assertions(model)

def test_init_ffn_from_bodyplan():
    """Test ffn is created properly from example_bodyplan
    """

    from crpm.ffn import FFN
    from crpm.ffn_bodyplan import read_bodyplan
    bodyplan = read_bodyplan("crpm/data/example_ffn_bodyplan.csv")

    #create FFN from bodyplan
    model = FFN(bodyplan)

    example_bodyplan_assertions(model)

def test_solve_ffn_numberadder():
    """test FFN class number adder can be solved by gradient decent
    """

    import numpy as np
    from crpm.ffn import FFN
    from crpm.dataset import load_dataset
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(4093082899)

    #create number adder from file
    model = FFN("crpm/data/numberadder_bodyplan.csv")

    #train numberadder model  with mean squared error
    _, data = load_dataset("crpm/data/numberadder.csv")
    _, _, _ = gradientdecent(model, data[0:5,], data[-1,], "mse",
                             healforces=False,
                             finetune=7)

    print(model.body[1]["weight"])

    assert np.allclose(model.body[1]["weight"], 1.0, rtol=.005)

def test_ffn_pre():
    """test static preprocessing block
    number adder should have negative weights with inverting pre-processor
    """

    import numpy as np
    from crpm.ffn import FFN
    from crpm.dataset import load_dataset
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(4093082899)

    #create an inverter for number adder model input
    inverter= FFN("crpm/data/numberadder_pre_bodyplan.csv")

    #assert inverter is 5 by 5 square
    assert inverter.body[1]["weight"].shape == (5, 5)

    #manually set weight to define the negative of the identity matrix
    inverter.body[1]["weight"] = -1*np.identity(5)

    #create number adder with inverter as pre-processor
    model = FFN("crpm/data/numberadder_bodyplan.csv", pre=inverter.body)

    #train numberadder model  with mean squared error
    _, data = load_dataset("crpm/data/numberadder.csv")
    _, _, _ = gradientdecent(model, data[0:5,], data[-1,], "mse", finetune=7)

    print(model.body[1]["weight"])

    assert np.allclose(model.body[1]["weight"], -1.0, rtol=.005)

def test_ffn_post():
    """test static post processing block
    number adder should have negative weights with inverting post-processor
    """

    import numpy as np
    from crpm.ffn import FFN
    from crpm.dataset import load_dataset
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(4093082899)

    #create an inverter for number adder model output
    inverter= FFN("crpm/data/numberadder_post_bodyplan.csv")

    #manually set weights to define the inverter
    inverter.body[1]["weight"] = np.array([[-1/2],[-1/2]])
    inverter.body[2]["weight"] = np.array([[1, 1]])

    #create number adder with inverter as pre-processor
    model = FFN("crpm/data/numberadder_bodyplan.csv", post=inverter.body)

    #train numberadder model  with mean squared error
    _, data = load_dataset("crpm/data/numberadder.csv")
    _, _, _ = gradientdecent(model, data[0:5,], data[-1,], "mse",
                             finetune=7)

    print(model.body[1]["weight"])

    assert np.allclose(model.body[1]["weight"], -1.0, rtol=.005)

def test_ffn_prepost():
    """test with both pre and post processing blocks
    number adder should have positive weights with both inverting pre and post
    processors
    """

    import numpy as np
    from crpm.ffn import FFN
    from crpm.dataset import load_dataset
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(4093082899)

    #create an inverter for number adder model input
    preinverter= FFN("crpm/data/numberadder_pre_bodyplan.csv")

    #assert inverter is 5 by 5 square
    assert preinverter.body[1]["weight"].shape == (5, 5)

    #manually set weight to define the negative of the identity matrix
    preinverter.body[1]["weight"] = -1*np.identity(5)

    #create an inverter for number adder model output
    postinverter= FFN("crpm/data/numberadder_post_bodyplan.csv")

    #manually set weights to define the inverter
    postinverter.body[1]["weight"] = np.array([[-1/2],[-1/2]])
    postinverter.body[2]["weight"] = np.array([[1, 1]])

    #create number adder with inverter as pre-processor
    model = FFN("crpm/data/numberadder_bodyplan.csv",
                pre=preinverter.body,
                post=postinverter.body)

    #train numberadder model  with mean squared error
    _, data = load_dataset("crpm/data/numberadder.csv")
    _, _, _ = gradientdecent(model, data[0:5,], data[-1,], "mse",
                             finetune=7)

    print(model.body[1]["weight"])

    assert np.allclose(model.body[1]["weight"], 1.0, rtol=.005)
