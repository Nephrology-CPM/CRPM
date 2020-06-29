""" Clustering by Kohonen self-organizing map
"""
def mapdata(map, state, data):
    """ computes node values for new features in matched data.

    Args:
        map: nfeature to nnode mapping
        state: nsamples of nfeatures
        data: matched nsamples of new features dim(nnewfeature, nobs)

    Returns: node values of dimension (nnode,nnewfeatures)
    """
    import numpy as np

    #get number of obs
    nobs = state[-1]["activity"].shape[1]
    #check number of samples in state and data are same
    if nobs != data.shape[1]:
        print("Error in mapdata: state and data do not have same number of observations")
        return None

    #get number of new features
    ndata = data.shape[0]

    #init mapped data
    mdata = np.zeros((ndata, map[-1]["n"]))
    mcount = np.zeros(map[-1]["n"])

    #diagnostic
    #print(nobs)
    #print(data.shape)
    #print(mdata.shape)

    #loop over samples
    for obv in range(nobs):
        #find closest node to observation
        node =np.argmin(np.linalg.norm(state[-2]["activity"][:, obv] -
                                       map[-1]["weight"],axis=1))
        #acumulate data
        mdata[:, node] += data[:, obv]
        mcount[node] += 1

    #normalize
    mdata[:,mcount>0] = mdata[:,mcount>0]/mcount[mcount>0]
    return mdata


def coords(n, nx=None, ny=None, hcp=None):
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
    return(xyz, np.array([[ix+1, jx+1, kx+1]]))

def init_som(model, state, n=100, nx=None, ny=None, hcp=False):
    """initializes a map from an ffn model

        Args:
                model: FFN model whose final layer is mapped
                n: number of mapping nodes default is 100
                nx: number of nodes in x direction
                ny: number of nodes in y direction
                hcp: boolean indicating use of hexagonal close packing default is False
    """

    import numpy as np
    from scipy.spatial import distance_matrix
    from crpm.ffn_bodyplan import get_bodyplan
    from crpm.ffn_bodyplan import init_ffn

    #make sure ffn top layer has logistic or softmax activation
    if(model[-1]["activation"]!="logistic" and model[-1]["activation"]!="softmax"):
        stop("som::init_map - input model is not a classifier.")

    #define number of clusters from size of top layer
    nclass = max(model[-1]["n"],2)

    #get model bodyplan
    bodyplan = get_bodyplan(model)

    #edit bodyplan toplayer to reflect number of mapping nodes and create map
    bodyplan[-1]["n"] = n
    bodyplan[-1]["activation"] = "gaussian"

    # create map
    map = init_ffn(bodyplan)

    #add node geometry to top layer and save unit cell scale factor
    map[-1]["coord"], scale = coords(n, nx, ny ,hcp)

    #calcualte node pair distances in mapping space for given geometry
    map[-1]["nodedist"] = distance_matrix(map[-1]["coord"],map[-1]["coord"])

    #multiply scale factor by 2 for unit radius
    scale = np.multiply(scale, 0.5)

    #initialize node weights based on
    #first 3 principal components of the penultimate layer activity
    #if data is multidimensional

    if state[-2]["activity"].shape[1]>1:
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

        #print(mact)
        #print(sig)
        #print(values)
        #print(vectors)

        #add zero vectors if number of features is less than 3
        if vectors.shape[0] < 3:
            zerovectors = np.zeros((3-vectors.shape[0],vectors.shape[1]))
            vectors = np.vstack((vectors,zerovectors))
            zerovectors = np.zeros((3-vectors.shape[0],1))
            #project node coordinates onto first 3 principal coordinates
            #unit scale coordinates then scale by feature stdev then translate to feature mean
            map[-1]["weight"] = ((map[-1]["coord"]/scale).dot(vectors[0:3,:]))*sig.T+mact[:, None].T

    return  map, nclass

def som(map, state, maxepoch=1000, lstart=1.0, lend=1E-8, nstart=2.0, nend=1E-3):#, mcclust=False ):
    """Train an som given the state of an ffn model

        Args:
            map: SOM initialized by FFN model
            state: FNN state by fwd propagation of input data
            maxepoch: max number of SOM training steps
            lstart: starting learning step size
            lend: ending learning step size
            nstart: strating neighborhood distance
            nend: ending neighborhood distance
        Returns: node coordinates mapped to each state sample
    """
    import copy
    import numpy as np
    import random
    from scipy.spatial import distance_matrix
    from progressbar import ProgressBar, Percentage, Bar, ETA, AdaptiveETA
    from crpm.activationfunctions import activation

    #get number of nodes and clusters
    nnode = map[-1]["n"]

    #iterate learning with exit conditions:
    # 1) too many iterations - hardcoded to ensure loop exit
    # 2) maxepoch is not positive signifies do no learning
    if maxepoch > 0:
        #set up for learning loop
        nobv = state[-2]["activity"].shape[1]
        #setup learning function and neighbor decay values ahead of loop
        lfunc = lstart*np.exp(-np.log(lstart/lend)/maxepoch*np.arange(maxepoch))
        sigma = nstart*np.exp(-np.log(nstart/nend)/maxepoch*np.arange(maxepoch))

        ##learning loop
        #widgets = [Percentage(),
        #           ' ', Bar(),
        #           ' ', ETA(),
        #           ' ', AdaptiveETA()]
        #pbar = ProgressBar(widgets=widgets)
        #for count in pbar(range(maxepoch)): #not really epochs, they are training steps
        for count in range(maxepoch): #not really epochs, they are training steps

            #shuffle sample order after each epoch
            if count%nobv == 0:
                sampleorder = np.arange(nobv)
                np.random.shuffle(sampleorder)

            #choose random sample
            #obv = np.random.randint(state[-2]["activity"].shape[1])
            obv = sampleorder[count%nobv]

            #calculate node vectors pointing to observation
            map[-1]["weightdot"] = state[-2]["activity"][:, obv] - map[-1]["weight"]

            #calcuate distances
            dist = np.linalg.norm(map[-1]["weightdot"],axis=1)

            #get winning node
            closestnode = np.argmin(dist)

            #calculate Neighborhood function
            nfunc = activation(map[-1]["activation"],
                            map[-1]["nodedist"][closestnode,:]/sigma[count]).reshape((nnode,1))

            #evolve nodes parameterized by winning node
            map[-1]["weight"] += lfunc[count]*nfunc*map[-1]["weightdot"]

    #calculate node pair distances in real space and weight by the map distance
    umat = distance_matrix(map[-1]["weight"],map[-1]["weight"])*map[-1]["nodedist"]
    #normalize umat assume boltzman-like distiribution
    umat = np.exp(-umat)
    #calculate mean of the inverse distances for each node: save umat in bias
    map[-1]["bias"] = np.mean(umat, axis=1, keepdims=True)
    return som_coords(map, state), map

def som_coords(map, state):
    """ return node coordinates for each observation"""
    import numpy as np

    #get number of obs
    nobs = state[-1]["activity"].shape[1]
    #set initial predictions all to -1
    pred = np.full((3, nobs), -1)
    #classify each observation
    for obv in range(nobs):
        #find closest node to observation
        node = np.argmin(np.linalg.norm(state[-2]["activity"][:, obv] -
                                        map[-1]["weight"],axis=1))
        #get cluster closest node belongs to
        pred[:, obv] = map[-1]["coord"][node]

    return pred


def som_nearestnode(map, state):
    """ return nearest node index for each observation"""
    import numpy as np

    #get number of obs
    nobs = state[-1]["activity"].shape[1]
    #set initial predictions all to -1
    pred = np.zeros(nobs)
    #classify each observation
    for obv in range(nobs):
        #find closest node to observation
        node = np.argmin(np.linalg.norm(state[-2]["activity"][:, obv] -
                                        map[-1]["weight"],axis=1))
        pred[obv] = node

    return pred
