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

    #mean center the coordinates
    xyz -= np.mean(xyz, axis=0)

    #return coordinates
    return(xyz, np.array([[ix+1,jx+1,kx+1]]))

def init_som(model, state, n=100, nx=None, ny=None, hcp=False):
    """initializes a map from an ffn model"""

    import numpy as np
    from scipy.spatial import distance_matrix
    from crpm.ffn_bodyplan import get_bodyplan
    from crpm.ffn_bodyplan import init_ffn

    #make sure ffn top layer has logistic or softmax activation
    if(model[-1]["activation"]!="logistic" and model[-1]["activation"]!="softmax"):
        stop("som::init_map - input model is not a classifier.")

    #define number of clusters from size top layer
    nclass = max(model[-1]["n"],2)

    #get model bodyplan
    bodyplan = get_bodyplan(model)

    #edit bodyplan toplayer to reflect number of mapping nodes and create map
    bodyplan[-1]["n"] = n
    bodyplan[-1]["activation"] = "gaussian"
    #bodyplan[-1]["regval"] =.01 #decay rate
    #bodyplan[-1]["lreg"] = 2 #init neighbor length

    # create map
    map = init_ffn(bodyplan)

    #add node geometry to top layer and save unit cell scale factor
    map[-1]["coord"],scale = coords(n, nx, ny ,hcp)

    #calcualte node pair distances in mapping space for given geometry
    map[-1]["nodedist"] = distance_matrix(map[-1]["coord"],map[-1]["coord"])

    #multiply scale factor by 2 for unit radius
    scale = np.multiply(scale,0.5)

    #init centroid locations close to origin
    map[-1]["centroid"] = np.random.random((nclass,3))

    #initialize node weights based on
    #first 3 principal components of the penultimate layer activity

    #define matrix with penultimate features in columns
    act = state[-2]["activity"]
    # calculate the mean of each feature
    mact = np.mean(act, axis=1)
    # mean center the features
    cact = act.T - mact
    # calculate covariance matrix of centered features
    vact = np.cov(cact.T)
    # eigendecomposition of covariance matrix
    values, vectors = np.linalg.eig(vact)
    #calcualte feature variance for scaling
    sig = np.std(act, axis=1)[:,None]
    #add zero vectors if number of features is less than 3
    if vectors.shape[0] < 3:
        zerovectors = np.zeros((3-vectors.shape[0],vectors.shape[1]))
        vectors = np.vstack((vectors,zerovectors))
        zerovectors = np.zeros((3-vectors.shape[0],1))
    #project node coordinates onto first 3 principal coordinates
    #unit scale coordinates then scale by feature stdev then translate to feature mean
    map[-1]["weight"] = ((map[-1]["coord"]/scale).dot(vectors[0:3,:]))*sig.T+mact[:, None].T

    return  map, nclass


def som(map, state, maxepoch=1000, lstart=1.0, lend=1E-8, nstart=2.0, nend=1E-3 ):
    """train som from an ffn model and data

        Args:
            map: SOM initialized by FFN model
            state: FNN state by fwd propagation of input data
            n: number of mapping nodes default is 100
            nx: number of nodes in x direction
            ny: number of nodes in y direction
            hcp: boolean indicating use of hexagonal close packing default is False
        Returns: final predictions and centroid coordinates in mapping space.
    """
    import numpy as np
    import random
    from scipy.spatial import distance_matrix
    from crpm.activationfunctions import activation

    #get number of nodes and clusters
    nnode = map[-1]["n"]
    nclass = map[-1]["centroid"].shape[0]

    #iterate learning with exit conditions:
    # 1) too many iterations - hardcoded to ensure loop exit
    # 2) maxepoch is not positive signifies do no learning
    if maxepoch > 0:
        #set up for learning loop
        count = 0
        continuelearning = True
        #setup learning function and neighbor decay values ahead of loop
        lfunc = lstart*np.exp(-np.log(lstart/lend)/maxepoch*np.arange(maxepoch))
        sigma = nstart*np.exp(-np.log(nstart/nend)/maxepoch*np.arange(maxepoch))
    else:
        #do no learning
        continuelearning = False
    #learning loop
    while continuelearning:

        #choose random sample
        obv = np.random.randint(state[-2]["activity"].shape[1])

        #calculate node vectors pointing to observation
        map[-1]["weightdot"] = state[-2]["activity"][:, obv] - map[-1]["weight"]

        #calcuate distances
        dist = np.linalg.norm(map[-1]["weightdot"],axis=1)

        #get winning node
        closestnode = np.argmin(dist)

        #calculate Neighborhood function
        #sigma = map[-1]["lreg"]*np.exp(-count*map[-1]["regval"])
        nfunc = activation(map[-1]["activation"],
                           map[-1]["nodedist"][closestnode,:]/sigma[count]).reshape((nnode,1))

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

    #calculate node pair distances in real space and inversely weight by the map distance
    umat = distance_matrix(map[-1]["weight"],map[-1]["weight"])/map[-1]["nodedist"]
    #accumulate weighted node distances for each node
    map[-1]["bias"] = np.nansum(umat,axis=1, keepdims=True)
    #normalize bias and assume boltzman-like distribution
    map[-1]["bias"] -= np.max(map[-1]["bias"])
    map[-1]["bias"] = np.exp(-map[-1]["bias"])
    map[-1]["bias"] /= np.sum(map[-1]["bias"])

    #make predicitons by k-means

    #assign initial centroid locations by dividing the n mapping node index.
    for icent in range(nclass):
        map[-1]["centroid"][icent,:] = map[-1]["coord"][nnode//nclass*icent,:]

    #k-means loop
    #iterate learning:
    # 1) too many iterations - hardcoded to ensure loop exit
    count = 0
    continuelearning = True
    while continuelearning:

        #calculate node distances to centroids
        dist = distance_matrix(map[-1]["coord"],map[-1]["centroid"])

        #find nearest centroid to each node
        nearestcentroid = np.argmin(dist, axis= 1)

        #init weighted center of mass
        cum = np.zeros((nclass,3))
        norm = np.zeros((nclass,1))

        #accumulate center of mass for nearest centroid
        for node in range(nnode):
            cum[nearestcentroid[node],:] += (map[-1]["coord"][node,:]
                                             *map[-1]["bias"][node])
            norm[nearestcentroid[node]] += map[-1]["bias"][node]

        #assign new centroid positions at the normalized center of mass
        map[-1]["centroid"] = cum/norm

        #update current learning step
        count += 1

        # - EXIT CONDITIONS -
        #exit if learning is taking too long
        if count >= int(maxepoch):
            print("Warning som.py: Tesselation is taking a long time!"+
                  " - Try increaseing maxepoch - Training will end")
            continuelearning = False

    #return predictions
    return som_classify(state, map), map

def som_classify(state, map):
    """ classify each observation using SOM """
    import numpy as np

    #get number of classes
    nclass = map[-1]["centroid"].shape[0]

    #get number of obs
    nobs = state[-1]["activity"].shape[1]

    #set initial predictions all to 0
    pred = np.zeros((1,nobs))

    #create a distance measure to each centroid
    dist = np.zeros(nclass)

    #classify each observation
    for obv in range(nobs):
        #find closest node to observation
        node =np.argmin(np.linalg.norm(state[-2]["activity"][:, obv] -
                                       map[-1]["weight"],axis=1))

        #find closest centroid to node
        for icent in range(nclass):
            #dist[icent] = np.linalg.norm(map[-1]["coord"][node,:]-centroid[icent,:])
            dist[icent] = np.linalg.norm(map[-1]["coord"][node,:]-
            map[-1]["centroid"][icent,:])
        closestnode = np.argmin(dist)

        #classify observation
        pred[0,obv] = closestnode

    return pred
