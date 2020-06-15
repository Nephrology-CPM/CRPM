""" test regularization on multicorrel_C data
"""
def test_regval_reduces_uncorrel():
    """ test that regularization term will reduce the weight assocciated with
    uncorrelated features compared with no regularization. Example function is
    y = 5x1 - x2 + 5x3^2, where features x1, x2, and x3 are indepently sampled from
    normal distribution with zero mean and unit variance."""

    import numpy as np
    from crpm.setup_multicorrel import setup_multicorrel
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(1500450271)

    #setup model with no regularization
    model, data = setup_multicorrel()

    #get data dimensions
    nobv = data.shape[1]

    # partition training and testing data
    train = data[0:3, 0:nobv//2]
    target = data[-1, 0:nobv//2]
    vtrain = data[0:3, nobv//2:nobv]
    vtarget = data[-1, nobv//2:nobv]

    #train model with mean squared error
    _, _, _ = gradientdecent(model, train, target, "mse", vtrain, vtarget,
                             finetune=8)

    #save weights
    weight0 = model[1]["weight"]

    #re-init model
    model, data = setup_multicorrel()

    #manually set regularization term
    model[1]["lreg"] = 1
    model[1]["regval"] = 100#75

    #train L1 regularized model
    _, _, _ = gradientdecent(model, train, target, "mse", vtrain, vtarget,
                             finetune=8)

    #save weights
    weight1 = model[1]["weight"]

    #re-init model
    model, data = setup_multicorrel()

    #manually set regularization term
    model[1]["lreg"] = 2
    model[1]["regval"] = 10

    #train L2 regularized model
    _, _, _ = gradientdecent(model, train, target, "mse", vtrain, vtarget,
                             finetune=8)

    #save weights
    weight2 = model[1]["weight"]

    assert abs(weight0[0, 2]) > abs(weight1[0, 2])
    assert abs(weight0[0, 2]) > abs(weight2[0, 2])

def test_regval_reduces_correl():
    """ test that regularization term will reduce the weight assocciated with
    correlated features compared with no regularization. Example function is
    y = x1 - x2 + x3^2, where features x1, x2, and x3 are normaly distributed
    with zero mean and unit variance and features x1 and x2 are correlated with
    value 0.5 while x3 is uncorrelated with both x1 and x2."""

    import numpy as np
    from crpm.ffn_bodyplan import reinit_ffn
    from crpm.setup_multicorrel import setup_multicorrel_c
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(1500450271)

    #setup model with no regularization
    model, data = setup_multicorrel_c()

    #get data dimensions
    nobv = data.shape[1]

    # partition training and testing data
    train = data[0:3, 0:nobv//2]
    target = data[-1, 0:nobv//2]
    vtrain = data[0:3, nobv//2:nobv]
    vtarget = data[-1, nobv//2:nobv]

    #train model with mean squared error
    _, _, _ = gradientdecent(model, train, target, "mse", vtrain, vtarget,
                             finetune=8)

    #save weights
    weight0 = model[1]["weight"]

    #switch to L1 regularization
    model[1]["lreg"] = 1

    #find regulaization value that minimizes cost by D&C
    #lmin = 5
    #lmax = 15
    ## Employ 8 rounds of D&C
    #for round in range(0,8):
    #    #get halfway point for regval
    #    alpha = (lmax+lmin)/2.0
    #    #LEFT HAND SIDE
    #    #re-init model
    #    model = reinit_ffn(model)
    #    #set regularization term to between lmin and alpha
    #    model[1]["regval"] = (lmin+alpha)/2.0
    #    #train L1 regularized model
    #    __, costl = gradientdecent(model, train, target, "mse",
    #                               validata=vtrain, valitargets=vtarget)
    #    #RIGHT HAND SIDE
    #    #re-init model
    #    model = reinit_ffn(model)
    #    #set regularization term to between lmax and alpha
    #    model[1]["regval"] = (lmax+alpha)/2.0
    #    #train L1 regularized model
    #    __, costr = gradientdecent(model, train, target, "mse",
    #                               validata=vtrain, valitargets=vtarget)
    #    #set new boundaries
    #    if costl < costr:
    #        lmax = alpha #set right hand boundary to alpha
    #    else:
    #        lmin = alpha #set left hand boundary to alpha

    #Calcualte L1 model between lmin and lmax
    #re-init model
    model = reinit_ffn(model)
    #set regularization term to between lmin and lmax
    model[1]["regval"] = 13.92578125#(lmin+lmax)/2.0
    #model[1]["regval"] = (lmin+lmax)/2.0
    #train L1 regularized model
    _, _, _ = gradientdecent(model, train, target, "mse", vtrain, vtarget,
                             finetune=8)

    #save weights and regval
    #alpha1 = model[1]["regval"]
    weight1 = model[1]["weight"]

    #switch to L2 regularization
    model[1]["lreg"] = 2

    #find regulaization value that minimizes cost by D&C
    #lmin = 0
    #lmax = 10
    ## Employ 8 rounds of D&C
    #for round in range(0,8):
    #    #get halfway point for regval
    #    alpha = (lmax+lmin)/2.0
    #    #LEFT HAND SIDE
    #    #re-init model
    #    model = reinit_ffn(model)
    #    #set regularization term to between lmin and alpha
    #    model[1]["regval"] = (lmin+alpha)/2.0
    #    #train L1 regularized model
    #    __, costl = gradientdecent(model, train, target, "mse",
    #                               validata=vtrain, valitargets=vtarget)
    #    #RIGHT HAND SIDE
    #    #re-init model
    #    model = reinit_ffn(model)
    #    #set regularization term to between lmax and alpha
    #    model[1]["regval"] = (lmax+alpha)/2.0
    #    #train L1 regularized model
    #    __, costr = gradientdecent(model, train, target, "mse",
    #                               validata=vtrain, valitargets=vtarget)
    #    #set new boundaries
    #    if costl < costr:
    #        lmax = alpha #set right hand boundary to alpha
    #    else:
    #        lmin = alpha #set left hand boundary to alpha

    #Calcualte L2 model between lmin and lmax
    #re-init model
    model = reinit_ffn(model)
    #set regularization term to between lmin and lmax
    model[1]["regval"] = 0.76171875#(lmin+lmax)/2.0
    #model[1]["regval"] = (lmin+lmax)/2.0
    #train L1 regularized model
    _, _, _ = gradientdecent(model, train, target, "mse", vtrain, vtarget,
                             finetune=9)

    #save weights and regval
    #alpha2 = model[1]["regval"]
    weight2 = model[1]["weight"]

    print("weights -------")
    print(weight0)
    print(weight1)
    print(weight2)
    #print("cost -------")
    #print(cost0)
    #print(cost1)
    #print(cost2)
    #print("reval -------")
    #print(alpha1)
    #print(alpha2)

    assert abs(weight0[0, 2]) > abs(weight1[0, 2])
    assert abs(weight0[0, 2]) > abs(weight2[0, 2])
    assert abs(weight2[0, 2]) != abs(weight1[0, 2])

'''
def test_calc_regval_dist():
    """ test that regval + sigma is yeilds a more parsimonious model.
    Example function is y = x1 - x2 + x3^2, where features x1, x2, and x3 are
    normaly distributed with zero mean and unit variance and features x1 and x2
    are correlated with value 0.5 while x3 is uncorrelated with both x1 and x2.
    """

    import numpy as np
    from crpm.setup_multicorrel import setup_multicorrel_c
    from crpm.ffn_bodyplan import reinit_ffn
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(1500450271)

    #setup model
    model, data = setup_multicorrel_c()
    #switch to L1 regularization
    model[1]["lreg"] = 1
    #define regval grid
    grid = range(0,20,1)

    #get data dimensions
    nvar = data.shape[0]
    nobv = data.shape[1]

    # partition data 80% training to 20% testing data
    test_data = data[0:3,0:nobv//5]
    test_target = data[-1,0:nobv//5]
    train_data = data[0:3,nobv//5:nobv]
    target_data = data[-1,nobv//5:nobv]

    #setup 5-fold cross validation on training data
    nfold = 5
    #get data dimensions
    nvar = train_data.shape[0]
    nobv = train_data.shape[1]
    #define cross-validation size according to number of observations
    invalid = np.zeros(nobv)
    #randomize training observations into n-folds
    shuffledobvs = np.arange(nobv)
    np.random.shuffle(shuffledobvs) #shuffle observation indecies in place
    k = 0
    for obv in shuffledobvs:
        invalid[obv] = k
        k = (k+1)%nfold

    #scan over regval grid
    mse = []
    for alpha in grid:

        #train model employing n-fold CV
        qinst = []
        for k in range(nfold):

            #reinit model
            model = reinit_ffn(model)
            #get data for this fold
            train = train_data[:, invalid != k]
            valid = train_data[:, invalid == k]
            #train model
            _, cost, _ = gradientdecent(model,train,target_data[invalid != k],
                                      "mse",valid,target_data[invalid == k])
            #save cost if well behaved
            if not (np.isnan(cost) or np.isinf(cost) or cost > 1E5):
                qinst.append(cost)
        #calculate mean MSE over n-folds for particular regval
        mse.append(np.mean(qinst))

    #print mse distribution
    print(mse)
    #calcualte probability distribution
    prob = np.exp(-(mse-np.min(mse)))
    norm = np.nansum(prob)
    prob = prob / norm
    print(prob)

    assert 1 == 2
'''

def test_deep_model():
    """ test that deep model has lower error than linear model. Example data is
    y = x1 - x2 + x3^2, where features x1, x2, and x3 are normaly distributed
    with zero mean and unit variance and features x1 and x2 are correlated with
    value 0.5 while x3 is uncorrelated with both x1 and x2."""

    import numpy as np
    from crpm.setup_multicorrel import setup_multicorrel_deep_c
    from crpm.setup_multicorrel import setup_multicorrel_c
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(1500450271)

    #setup shallow model
    model, data = setup_multicorrel_c()

    #get data dimensions
    nobv = data.shape[1]

    # partition training and testing data
    train = data[0:3, 0:nobv//2]
    target = data[-1, 0:nobv//2]
    vtrain = data[0:3, nobv//2:nobv]
    vtarget = data[-1, nobv//2:nobv]

    #train model with mean squared error
    _, cost0, _ = gradientdecent(model, train, target, "mse", vtrain, vtarget,
                                 finetune=8)

    #save weights
    #weight0 = model[1]["weight"]

    #setup deep model
    model, data = setup_multicorrel_deep_c()

    #train model with mean squared error
    _, cost1, _ = gradientdecent(model, train, target, "mse", vtrain, vtarget,
                                 finetune=8)

    #save weights
    #weight1 = model[1]["weight"]

    #print(cost0)
    #print(cost1)
    #print(weight0)
    #print(weight1)
    #print(model[2]["weight"])

    assert cost0 > cost1
