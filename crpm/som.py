""" Clustering by Kohonen self-organizing map
"""
def mapdata(map, state, data):
    """ computes node values for new features in matched data.

    Args:
        map: nfeature to nnode mapping
        state: nsamples of nfreatures
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

    #init node classes to zero indicating all in the same cluster
    map[-1]["cluster"] = np.zeros(n, dtype=int)

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


def som(map, state, maxepoch=1000, lstart=1.0, lend=1E-8, nstart=2.0, nend=1E-3, mcclust=False ):
    """train som from an ffn model and data

        Args:
            map: SOM initialized by FFN model
            state: FNN state by fwd propagation of input data
            n: number of mapping nodes default is 100
            nx: number of nodes in x direction
            ny: number of nodes in y direction
            hcp: boolean indicating use of hexagonal close packing default is False
        Returns: final predictions and centroid coordinates in mapping space.
        TODO: include nsmooth(int) and clusterby(string 'MC', 'knn', or 'greedy') args
    """
    import copy
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

    #calculate node pair distances in real space and weight by the map distance
    umat = distance_matrix(map[-1]["weight"],map[-1]["weight"])*map[-1]["nodedist"]
    #normalize umat assume boltzman-like distiribution
    umat = np.exp(-umat)
    #calculate mean of the inverse distances for each node
    map[-1]["bias"] = np.mean(umat, axis=1, keepdims=True)

    #Repeat convolutional smoothing
    nsmooth = 1
    if nsmooth > 0:
        for iconvo in range(nsmooth):
            #init convolution
            convo = np.zeros(map[-1]["bias"].shape)
            #loop over nodes
            for node in range(nnode):
                #get neigboring nodes at least 1.5 unit distance away
                neighborhood = np.where(map[-1]["nodedist"][node,:]<=1.5)[0]
                #get average umat value
                convo[node] = np.mean(map[-1]["bias"][neighborhood])
            #retrun smoothed Umatrix
            map[-1]["bias"] = copy.deepcopy(convo)

    # ---initial clustering---
    # assign each node its own trajectory and
    # each trajectory follows umat graident to local maximum
    for startnode in range(nnode):
        #init centroid at starting node
        traj = copy.copy(startnode)
        followinggradient = True
        while followinggradient:
            #get neigboring nodes at least 1.5 unit distance away
            neighborhood = np.where(map[-1]["nodedist"][traj,:]<=1.5)[0]
            #find node in neiborhood with with highest umat
            minnode = neighborhood[np.argmax(map[-1]["bias"][neighborhood])]
            #stop if at minimum
            if minnode == traj:
                followinggradient = False
            #update trajectory location
            traj = minnode
        #assign local minimum to starting node
        map[-1]["cluster"][startnode] = np.copy(traj)

    # get number of local minima (initial clusters)
    cstate = np.unique(map[-1]["cluster"]).astype(int)
    nclass = len(cstate)
    cluster_copy = np.copy(map[-1]["cluster"])
    # Assign each cluster its own state
    for iclust in range(nclass):
        #get nodes in cluster
        inclust = np.where(map[-1]["cluster"]==cstate[iclust])[0]
        cluster_copy[inclust]=iclust
    #reassign clusters
    cstate = np.arange(nclass)
    map[-1]["cluster"]=np.copy(cluster_copy)
    #print(nclass)

    #Minimize Davies-Bouldin (DB) metric
    #(future options by None, montecarlo, knn, or greedy algorithm)
    if mcclust:
        #assign state by montecarlo minimization
        cstate = montecarlo(map, cstate, maxstep=5000, adjstep=50)
        #cstate = greedymin(map, cstate)
        #print(cstate)


    #    ####
    #make predicitons by k-means
    ##assign initial centroid locations by dividing the n mapping node index.
    #for icent in range(nclass):
    #    map[-1]["centroid"][icent,:] = map[-1]["coord"][nnode//nclass*icent,:]
    #k-means loop
    #iterate learning:
    # 1) too many iterations - hardcoded to ensure loop exit
    #count = 0
    #continuelearning = True
    #while continuelearning:
    #    #calculate node distances to centroids
    #    dist = distance_matrix(map[-1]["coord"],map[-1]["centroid"])
    #    #find nearest centroid to each node
    #    nearestcentroid = np.argmin(dist, axis= 1)
    #    #init weighted center of mass
    #    cum = np.zeros((nclass,3))
    #    norm = np.zeros((nclass,1))
    #    #accumulate center of mass for nearest centroid
    #    for node in range(nnode):
    #        cum[nearestcentroid[node],:] += (map[-1]["coord"][node,:]
    #                                         *map[-1]["bias"][node])
    #        norm[nearestcentroid[node]] += map[-1]["bias"][node]
    #    #assign new centroid positions at the normalized center of mass
    #    map[-1]["centroid"] = cum/norm
    #    #update current learning step
    #    count += 1
    #    # - EXIT CONDITIONS -
    #    #exit if learning is taking too long
    #    if count >= int(maxepoch):
    #        print("Warning som.py: Tesselation is taking a long time!"+
    #              " - Try increaseing maxepoch - Training will end")
    #        continuelearning = False
    #    ####



    #reassign clusters based on new cstate
    cluster_copy = np.copy(map[-1]["cluster"])
    # Assign each cluster its own state
    for iclust in range(nclass):
        #get nodes in cluster
        inclust = np.where(map[-1]["cluster"]==iclust)[0]
        cluster_copy[inclust]=cstate[iclust]
    #reassign clusters
    map[-1]["cluster"]=np.copy(cluster_copy)

    # reorder unique clusters
    cstate = np.unique(map[-1]["cluster"]).astype(int)
    nclass = len(cstate)
    cluster_copy = np.copy(map[-1]["cluster"])
    # Assign each cluster its own state
    for iclust in range(nclass):
        #get nodes in cluster
        inclust = np.where(map[-1]["cluster"]==cstate[iclust])[0]
        cluster_copy[inclust]=iclust
    #reassign clusters
    cstate = np.arange(nclass)
    map[-1]["cluster"]=np.copy(cluster_copy)

    #find centroid locations
    ncent = len(np.unique(cstate).astype(int))

    map[-1]["centroid"] = np.zeros((ncent,3))
    #get centroid center of mass
    for icent in range(ncent):
        #get nodes in cluster
        inclust = np.where(map[-1]["cluster"] == icent)[0]
        #center of mass
        map[-1]["centroid"][icent] = np.mean(map[-1]["coord"][inclust], axis=0)

    #return predictions
    return som_classify(map, state), map

def som_classify(map, state):
    """ classify each observation using SOM """
    import numpy as np

    #get number of obs
    nobs = state[-1]["activity"].shape[1]
    #set initial predictions all to -1
    pred = np.full((1, nobs), -1)
    #classify each observation
    for obv in range(nobs):
        #find closest node to observation
        node = np.argmin(np.linalg.norm(state[-2]["activity"][:, obv] -
                                        map[-1]["weight"],axis=1))
        #get cluster closest node belongs to
        pred[0, obv] = map[-1]["cluster"][node]

    return pred

def davies_bouldin_metric(map, cstate):
    """ Calcualtes the Davies Bouldin Metric for a specific cluster assigment.
    """
    import numpy as np
    from scipy.spatial import distance_matrix

    #get number of classes
    ustate = np.unique(cstate)
    nclass = len(ustate)

    #get cluster: members, centroids, and scatter
    inclust = [] #init nodes list per cluster
    centroid = [] #init centroid list
    scatter = []
    for iclust in ustate:
        #find cstates with same ustate
        clusters = np.where(cstate == iclust)[0]
        inclust.append(np.in1d(map[-1]["cluster"],clusters))#get nodes in all clusters with ustate
        #inclust.append(np.where(map[-1]["cluster"]==ustate[iclust])[0]) #get nodes in cluster
        if len(inclust[-1]) > 1:
            #get centroid (center of mass)
            centroid.append(np.mean(map[-1]["weight"][inclust[-1],], axis=0))
            #compute within-cluster distance
            #sint = np.sqrt(np.mean(np.sum(np.square(map[-1]["weight"][inclust,]-centroid[-1]), axis=1)))
            cdata = map[-1]["weight"][inclust[-1]]-centroid[-1]
            scatter.append(np.sqrt(np.mean(np.sum(np.square(cdata), axis=1))))

    ## get intercluster distance
    smat = distance_matrix(centroid,centroid)

    #get db metric
    dbmat = np.full(smat.shape,np.nan)
    for iclust in range(nclass-1):
        for jclust in range(iclust+1, nclass):
            dbmat[iclust, jclust] = (scatter[iclust]+scatter[jclust])/smat[iclust, jclust]
            dbmat[jclust, iclust] = dbmat[iclust, jclust]

    return np.nanmean(np.nanmax(dbmat, axis=0))

def transition(cstate):
    """ Application of detailed balance. Method randomly chooses one state and
    assigns it random value. All transitions are equally proabable.
    """
    import numpy as np

    #pick random state
    idx = np.random.choice(len(cstate))
    #assign that random state a new value
    cstate[idx] = np.random.choice(len(cstate))
    return cstate

def montecarlo(map, cstate, maxstep, adjstep):
    """ montecarlo sampling based on Davies-Bouldin (DB) metric"""
    import copy
    import numpy as np

    #init Q difference
    burninstep = adjstep*2
    qdif = np.full(burninstep,np.nan)
    #get current Q
    currq = np.mean(davies_bouldin_metric(map, cstate))
    for istep in range(burninstep):
        ## sample Q
        transition(cstate)
        trialq = np.mean(davies_bouldin_metric(map, cstate))
        #accumulate Q difference
        qdif[istep] = abs(trialq - currq)
        #replace current Q
        currq = copy.deepcopy(trialq)
    #get average Q difference
    qdif = np.nanmean(qdif)
    #estimate initial beta with mean Q difference
    beta = -1*np.log(.1)/qdif
    #print("initial beta")
    #print(beta)
    #print(qdif)
    #print("---")

    #save initial(best) state and Q
    currq = np.mean(davies_bouldin_metric(map, cstate))
    best = copy.deepcopy(cstate)
    bestq = copy.deepcopy(currq)

    #init MC step and acceptance count
    istep = 0
    account = 0
    #main MC loop
    while istep < maxstep:
        ## increment step
        istep += 1
        ## get trial state and Q
        trial = copy.deepcopy(cstate)
        transition(trial)
        trialq = np.mean(davies_bouldin_metric(map, trial))
        ## accept/reject trial state
        if np.random.uniform() < np.exp(-beta*(trialq-currq)): # pylint: disable=comparison-with-callable
            #accept trial
            cstate = copy.deepcopy(trial)
            currq = copy.deepcopy(trialq)
            #increment acceptance counter
            account += 1
        ## save best state and Q
        if currq <= bestq:
            best = copy.deepcopy(cstate)
            bestq = copy.deepcopy(currq)
        ## periodically adjust beta
        if istep%adjstep == 0:
            ##Diagnostic
            #print(istep)
            #print(account/adjstep)
            #print(beta)
            #print(currq)
            #print(bestq)
            #print(best)
            #print(len(np.unique(best)))
            #print("---")
            #decrease beta (increase temp) if acceptance ratio is less than half
            #  else increase beta
            if account/adjstep < 0.5:
                beta *= .95
            else:
                beta *= 1.05
                #reset acceptance counter
                account = 0

    #set cstate to best partition
    cstate = copy.deepcopy(best)

    return best

def greedymin(map, cstate):
    """ greedy algorithm to minimize Davies-Bouldin (DB) metric"""
    import copy
    import numpy as np

    #save initial(best) state and Q
    currq = np.mean(davies_bouldin_metric(map, cstate))
    best = copy.deepcopy(cstate)
    bestq = copy.deepcopy(currq)
    nstate = len(cstate)

    #greedy algorithm
    followinggradient = True
    ntrial = 0
    while followinggradient:
        ntrial += 1
        ## loop over trial states
        for istate in range(nstate):
            for jstate in range(nstate):
                #increment trial counter
                ntrial += 1
                #reset trial state to current state
                trial = copy.deepcopy(cstate)
                #get trial state
                trial[istate] = (trial[istate]+jstate)%nstate
                #calculate trial state metric
                trialq = np.mean(davies_bouldin_metric(map, trial))
                ## save best state and Q
                if trialq <= bestq:
                    best = copy.deepcopy(trial)
                    bestq = copy.deepcopy(trialq)
        #update current state to best state
        if bestq<currq:
            cstate = copy.deepcopy(best)
            currq = copy.deepcopy(bestq)
        else:
            followinggradient = False

        #stop greedy algorithim if only 1 state exists
        if len(np.unique(cstate))<2:
            followinggradient = False

        ##Diagnostic
        #print(ntrial)
        #print(currq)
        #print(cstate)
        #print("---")

    return cstate
