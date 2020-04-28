""" test self oranizing map algorithm
"""

def test_som1d_init_pca():
    """test inital node coordinates for 1D SOM follow known PCA for nested Cs data.

    We know for Nested Cs data - first 2 PCs point roughly in x and y directions
    """

    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.som import init_som
    from crpm.som import som

    #init numpy seed
    np.random.seed(40017)

    #setup model with nested Cs
    model, data = setup_nestedcs()

    #update state of model
    _, state = fwdprop(data[0:2,], model)

    #create and init 1D map with model and its current state
    map, _ = init_som(model, state)

    #check nodes map to points along x direction with reasonable span
    xmin = min(data[0,])
    xmax = max(data[0,])
    wmin = min(map[-1]["weight"][:,0])
    wmax = max(map[-1]["weight"][:,0])
    xvar = np.var(data[0,])
    #assert nodes point in the correct x direction
    assert(np.sign(wmax-wmin)==np.sign(xmax-xmin))
    #asset span of x points is comperable to span of input x points
    assert(abs(np.var(map[-1]["weight"][:,0])-xvar)/xvar < 1.0)
    #assert node points have no (little) y component
    assert(np.var(map[-1]["weight"][:,1]) < 0.1)

def test_som2d_init_pca():
    """test inital node coordinates for 2D SOM follow known PCA for nested Cs data.

    We know for Nested Cs data - first 2 PCs point roughly in x and y directions
    """

    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.som import init_som
    from crpm.som import som

    #init numpy seed
    np.random.seed(40017)

    #setup model with nested Cs
    model, data = setup_nestedcs()

    #update state of model
    _, state = fwdprop(data[0:2,], model)

    #create and init 2D map with model and its current state
    map, _ = init_som(model, state, n=100, nx=10, ny=10, hcp=True)

    #check nodes map to points along x and y direction with reasonable span
    xmin = min(data[0,])
    xmax = max(data[0,])
    xvar = np.var(data[0,])
    wxmin = min(map[-1]["weight"][:,0])
    wxmax = max(map[-1]["weight"][:,0])
    wxdel = map[-1]["weight"][-1,0] - map[-1]["weight"][0,0]
    ymin = min(data[1,])
    ymax = max(data[1,])
    yvar = np.var(data[1,])
    wymin = min(map[-1]["weight"][:,1])
    wymax = max(map[-1]["weight"][:,1])
    wydel = map[-1]["weight"][-1,1] - map[-1]["weight"][0,1]
    slope = (ymax-ymin)/(xmax-xmin)

    #assert nodes point in the correct x direction
    assert(np.sign(wxmax-wxmin)==np.sign(xmax-xmin))
    #assert nodes point in the correct y direction
    assert(np.sign(wymax-wymin)==np.sign(ymax-ymin))
    #assert fist and last nodes point with slope similar to data range
    assert(abs(wydel/wxdel - slope)/slope < 0.4 )
    #asset span of x points is comperable to span of input x points
    assert(abs(np.var(map[-1]["weight"][:,0])-xvar)/xvar < 1.0)
    #asset span of y points is comperable to span of input y points
    assert(abs(np.var(map[-1]["weight"][:,1])-yvar)/yvar < 1.0)


def r_test_solve_nestedcs():
    """test nested cs can be solved
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.som import init_som
    from crpm.som import som
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_nestedcs()

    #update state of model
    _, state = fwdprop(data[0:2,], model)

    #create and init map with model and its current state
    map, _ = init_som(model, state, n=100, nx=100, ny=1, hcp=True)

    #plot data and map in real space
    plt.scatter(data[0,],data[1,], c=data[-1,])
    plt.plot(map[-1]["weight"][:,0],map[-1]["weight"][:,1])
    plt.scatter(map[-1]["weight"][:,0],map[-1]["weight"][:,1])
    plt.show()

    #conduct mapping
    pred, map = som(map, state, lstart = .2, lend=.001, nstart=50.0, nend=0.001, maxepoch=5000)

    #plot data and map in real space
    plt.scatter(data[0,],data[1,], c=data[-1,])
    plt.plot(map[-1]["weight"][:,0],map[-1]["weight"][:,1])
    plt.scatter(map[-1]["weight"][:,0],map[-1]["weight"][:,1])
    plt.show()

    #plot map and centroids in mapping space
    plt.scatter(map[-1]["coord"][:,0],map[-1]["coord"][:,1],
                c=map[-1]["bias"][:,0], cmap='gray')
    #plt.scatter(map[-1]["centroid"][:,0],map[-1]["centroid"][:,1], s=100)
    plt.show()

    #plot predictions in real space
    #plt.scatter(data[0,], data[1,], c=data[-1,]-pred[0,])
    #plt.show()

    #analyze binary classifier
    _, report = analyzebinaryclassifier(pred, data[-1,])
    print(report)

    assert report["MatthewsCorrCoef"] >= .5
