""" Clustering by Kohonen self-organizing map
"""

def coords(n,nx=None,ny=None,hcp=None):
    """ returns the coordinates of n points in 3 dimensions
        Args:
            n: integer number of coordinates to return
            nx: length of cell in x direction
            ny: length of cell in y direction
            hcp: boolean if True will arrange in hexagonal close packed else
                    will arrange in cubic
        Returns:
            an numpy array of 3D coordinates where the 3rd dimension is zero.
            The function returns 3D coordinates to conform with som format.
    """
    import math
    import numpy as np
    # get number of nodes in x direction (defalut is linear arrangement)
    if nx == None:
        nx = n

    # get number of nodes in y direction (default is linear arrangement)
    if ny == None:
        ny = 1

    #some konstants
    root3 = math.sqrt(3)
    third = 1/3
    twothirdroot6 = 2*third*math.sqrt(6)

    #init coordinate array
    xyz = np.zeros((n,3))

    #loop over nodes
    node = 0
    while node < n:
        ix = node%nx
        jx = (node//nx)%ny
        kx = node//(nx*ny)
        #get hcp coordinates
        if hcp:
            xyz[node,0] = 2*ix+(jx+kx)%2
            xyz[node,1] = root3*(jx+third*(kx%2))
            xyz[node,2] = twothirdroot6*kx
            xyz[node,:] /= 2
        else: #get cubic coordinates
            xyz[node,0] = ix
            xyz[node,1] = jx
            xyz[node,2] = kx
        #increment node
        node += 1

    #return coordinates
    return(xyz)

def init_map(model, n=100, nx=None, ny=None, hcp=False):
    """initializes a map from an ffn model"""
    from crpm.ffn_bodyplan import get_bodyplan
    from crpm.ffn_bodyplan import init_ffn

    #make sure ffn top layer has logistic or softmax activation
    if(model[-1]["activation"]!="logistic" and model[-1]["activation"]!="softmax"):
        stop("som::init_map - input model is not a classifier.")

    #define number of clusters from model top layer
    nclass = model[-1]["n"]

    #get model bodyplan
    bodyplan = get_bodyplan(model)

    #edit bodyplan toplayer to reflect number of mapping nodes and create map
    bodyplan[-1]["n"] = n
    bodyplan[-1]["activation"] = "gaussian"
    bodyplan[-1]["regval"] =.01 #decay rate
    bodyplan[-1]["lreg"] = 2 #init neighbor length

    # create map
    map = init_ffn(bodyplan)

    #add node geometry to top layer
    map[-1]["coord"] = coords(n, nx, ny ,hcp)

    return  map, nclass


def som(model, data, n=100, nx=None, ny=None, hcp=False):
    """train fnn model by gradient decent

        Args:
            model: FFN model
            data: input features
            targets: targets
            geom: map topology either "square",...
            n: number of mapping nodes default is 10000
        Returns: final predictions and converged map.
    """
    import numpy as np
    import random
    from crpm.fwdprop import fwdprop
    from crpm.activationfunctions import activation

    #define hyperparameters
    maxepoch = 1000

    #setup Learning function
    lfunc = np.exp(-np.log(1E-12)/maxepoch*np.arange(maxepoch))

    #create map from model
    map, nclass = init_map(model, n, nx, ny, hcp)

    #set initial predictions all to 0
    pred = np.zeros((nclass,data.shape[1]))

    #update state of model and get model predictions
    modelpred, state = fwdprop(data, model)

    #iterate learning:
    # 1) too many iterations - hardcoded to ensure loop exit
    count = 0
    continuelearning = True
    while continuelearning:

        #choose random sample
        obv = np.random.randint(data.shape[1])

        #calculate node vectors pointing to observation
        map[-1]["weightdot"] = state[-2]["activity"][:, obv] - map[-1]["weight"]

        #calcuate distances
        dist = np.linalg.norm(map[-1]["weightdot"],axis=1)

        #get winning node
        closestnode = np.argmin(dist)

        #calculate Neighborhood function
        sigma = map[-1]["lreg"]*np.exp(-count*map[-1]["regval"])
        nfunc = activation(map[-1]["activation"], dist/sigma).reshape((n,1))

        #evolve nodes parameterized by winning node
        map[-1]["weight"] += lfunc[count]*nfunc*map[-1]["weightdot"]

        #update current learning step
        count += 1

        # - EXIT CONDITIONS -
        #exit if learning is taking too long
        if count >= int(maxepoch):
            print("Warning som.py: Training is taking a long time!"+
                  " - Try increaseing maxepoch - Training will end")
            continuelearning = False

    #calcualate umatrix and save in bias
    for node in range(n):
        cum = 0
        for inode in range(n):
            if node != inode:
                cum += (np.linalg.norm(map[-1]["weight"][inode,:]-map[-1]["weight"][node,:])/
                       np.linalg.norm(map[-1]["coord"][inode,:]-map[-1]["coord"][node,:]))

        map[-1]["bias"][node] = cum

    #normalize cumulative and assume boltzman-like distribution
    cum = map[-1]["bias"] / np.max(map[-1]["bias"])
    cum = np.exp(-cum)
    cum /= np.sum(cum)
    map[-1]["bias"] = cum

    #import csv
    #file = open("test.csv", "w")
    #with file:
    #    writer = csv.writer(file)
    #    writer.writerows(map[-1]["bias"])

    #make predicitons by k-means
    binary = False
    if nclass == 1: #check for binary classification
        nclass = 2
        binary = True

    centroid = np.zeros((nclass,3)) #init centroids
    #randomly assign nclusters using the n mapping node coordinates.
    cidx = np.random.randint(n,size=nclass)
    for icent in range(nclass):
        centroid[icent,:] = map[-1]["coord"][cidx[icent],:]

    #create a distance measure for each centroid
    dist = np.zeros(nclass)

    #k-means loop
    #iterate learning:
    # 1) too many iterations - hardcoded to ensure loop exit
    count = 0
    continuelearning = True
    while continuelearning:

        #init weighted center of mass
        cum = np.zeros((nclass,3))
        norm = np.zeros((nclass,1))

        #accumulate center of mass for closest centroid
        for node in range(n):
            #get distance to each centroid
            for icent in range(nclass):
                dist[icent] = np.linalg.norm(map[-1]["coord"][node,:]-centroid[icent,:])
            #find closest centroid
            closestnode = np.argmin(dist)
            #calculate the center of mass weighted by the boltzman prob saved in bias
            cum[closestnode,:] += map[-1]["coord"][node,:]*map[-1]["bias"][node]
            norm[closestnode] += map[-1]["bias"][node]

        #assign new centroid positions at the normalized center of mass
        centroid = cum/norm

        #update current learning step
        count += 1

        # - EXIT CONDITIONS -
        #exit if learning is taking too long
        if count >= int(maxepoch):
            print("Warning som.py: Tesselation is taking a long time!"+
                  " - Try increaseing maxepoch - Training will end")
            continuelearning = False

    #classify each observation
    for obv in range(data.shape[1]):
        #find closest node to observation
        node =np.argmin(np.linalg.norm(state[-2]["activity"][:, obv] -
                                       map[-1]["weight"],axis=1))

        #find closest centroid to node
        for icent in range(nclass):
            dist[icent] = np.linalg.norm(map[-1]["coord"][node,:]-centroid[icent,:])
        #find closest centroid
        closestnode = np.argmin(dist)

        #classify observation
        pred[:,obv] = 0
        if binary:
            pred[0,obv] = 1-closestnode
        else:
            pred[closestnode,obv] = 1


    #return predictions and converged map
    return pred, map
