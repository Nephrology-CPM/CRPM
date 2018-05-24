"""Test deep neural network body plan functions.
"""

def test_example_bodyplan():
    """Check example_bodyplan.csv is as we expect
    """

    from crpm.dnn_bodyplan import read_bodyplan
    body_plan = read_bodyplan("crpm/data/example_dnn_bodyplan.csv")

    assert body_plan.shape[0] == 5
    assert body_plan.shape[1] == 3

    assert body_plan.loc[0, 'n'] == 2
    assert body_plan.loc[1, 'n'] == 3
    assert body_plan.loc[2, 'n'] == 5
    assert body_plan.loc[3, 'n'] == 7
    assert body_plan.loc[4, 'n'] == 1

    assert body_plan.loc[0, 'activation'] == 'identity'
    assert body_plan.loc[1, 'activation'] == 'relu'
    assert body_plan.loc[2, 'activation'] == 'relu'
    assert body_plan.loc[3, 'activation'] == 'relu'
    assert body_plan.loc[4, 'activation'] == 'logistic'
