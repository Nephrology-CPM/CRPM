""" test early stopping various learning algorithms"""

#early stopping constant
NOBV = 22

def test_earlystopping_triggered():
    """ test early stopping is triggered with overfitting dataset."""

    import numpy as np
    from crpm.setup_overfitting import setup_overfitting_shallow
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(40017)

    #setup shallow model
    model, _, train, valid = setup_overfitting_shallow()
    #print(train.shape)
    #print(valid.shape)
    train = train[:, :NOBV]

    #assert early stopping is triggered with dataset
    _, _, ierr = gradientdecent(model, train[:-1, :], train[-1, :],
                                "mse", valid[:-1, :], valid[-1, :], earlystop=True)

    #does early stopping message appears
    #assert True
    print(ierr)
    assert ierr == 2

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
    train = train[:, :NOBV] #reduce training data to ensure early stopping occours

    #calculate initial error
    _, cost0, _ = gradientdecent(model, train[:-1, :], train[-1, :],
                                 "mse", valid[:-1, :], valid[-1, :], maxepoch=0)

    #reinit model
    model = reinit_ffn(model)

    #calculate out-sample error with no early stopping
    start_time = time.clock()
    _, cost, _ = gradientdecent(model, train[:-1, :], train[-1, :],
                                "mse", valid[:-1, :], valid[-1, :])
    run_time = time.clock()-start_time

    #reinit model
    model = reinit_ffn(model)

    #calculate out-sample error with early stopping
    start_time = time.clock()
    _, cost2, _ = gradientdecent(model, train[:-1, :], train[-1, :],
                                 "mse", valid[:-1, :], valid[-1, :], earlystop=True)
    run_time2 = time.clock() - start_time

    #calculate learning efficiency defined as amount cost reduced per wall-clock time.
    leff = (cost0-cost)/run_time
    leff2 = (cost0-cost2)/run_time2

    print("cost initial = "+str(cost0))
    print("cost = "+str(cost))
    print("cost earlystop = "+str(cost2))
    print("runtime = "+str(run_time))
    print("runtime earlystop = "+str(run_time2))
    print("efficiency = "+str((cost0-cost)/run_time))
    print("efficiency earlystop= "+str((cost0-cost2)/run_time2))

    #assert learning with early stopping is faster than without
    assert run_time2 < run_time

    #assert cost with early stopping is higher than without
    assert cost2 > cost

    #assert relative difference in learning efficiency is less than .2 with earlystopping
    assert abs(leff2-leff)/leff < .2
