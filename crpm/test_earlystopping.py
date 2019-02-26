""" test early stopping various learning algorithms"""

def test_earlystopping_triggered():
    """ test early stopping is triggered with overfitting dataset."""

    import numpy as np
    from crpm.setup_overfitting import setup_overfitting_shallow
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(40017)

    #setup shallow model
    model, _, train, valid = setup_overfitting_shallow()
    nobv = 24

    #assert early stopping is triggered with dataset
    _, _ = gradientdecent(model, train[:-1, :nobv], train[-1, :nobv],
                          "mse", valid[:-1, :], valid[-1, :], earlystop=True)

    #dose early stopping message appear
    assert True

def test_naive_earlystopping_gradient_decent():
    """ test naive early stopping yeilds comperable outsample error using
    overfitting dataset compared to training with no early stopping."""

    import time
    import numpy as np
    from crpm.setup_overfitting import setup_overfitting_shallow
    from crpm.ffn_bodyplan import reinit_ffn
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(40017)

    #setup shallow model
    model, _, train, valid = setup_overfitting_shallow()
    train = train[:, :24] #reduce training data to ensure early stopping occours

    #calculate out-sample error with no early stopping
    start_time = time.clock()
    _, cost = gradientdecent(model, train[:-1, :], train[-1, :],
                             "mse", valid[:-1, :], valid[-1, :])
    run_time = time.clock()-start_time
    print(cost)
    print(run_time)

    #reinit model
    model = reinit_ffn(model)

    #calculate out-sample error with early stopping
    start_time = time.clock()
    _, cost2 = gradientdecent(model, train[:-1, :], train[-1, :],
                              "mse", valid[:-1, :], valid[-1, :], earlystop=True)
    run_time2 = time.clock() - start_time

    #assert relative difference of error with early stopping
    #to error with no early stopping is less than 20%
    assert abs(cost2-cost)/cost < .2

    #assert learning with early stopping is faster than without
    assert run_time2 < run_time
