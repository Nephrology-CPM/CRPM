"""Implements Gillespie stochastic simulation algorithm for chemical kinetics"""

def ssa(state, nmat, kvec, runtime=0, clamp=None):
    """ Gillespie SSA will simulate the evolution of N chemcial species given
    M chemical reactions with rates constants defined.

        Args
            state: real array of N initial chemical species initial populations
            nmat: integer matrix of dimension (N,M) defining reaction stoichiometry
            kvec: real array of M chemical reaction rate constants
    """

    import numpy as np

    #init exit error condition
    ierr = 0

    #state defines N species
    nspec = len(state)

    #kvec defines M reactions
    nrxn = len(kvec)

    #convert input to numpy arrays
    state = np.array(state)
    kvec = np.array(kvec)
    nmat = np.array(nmat)

    #reshape vectors
    kvec = np.reshape(kvec, (nrxn, 1))
    state = np.reshape(state, (nspec, 1))

    #define poulation change matrix
    dstate = np.subtract(nmat[nspec:, :], nmat[:nspec, :])

    #apply clamp where defined
    if clamp is not None:
        #check clamp is defined for each Species
        if len(clamp) == nspec:
            dstate[np.where(clamp), :] = 0
        else:
            print("Warning: clamp does not conform to number of species.")
            print("Warning: ignoring clamp.")
            #indicate warning with ierr
            ierr = 1

    #init trajectory
    time = 0
    trajectory = np.vstack((state, time))

    #assert full stochiometry matrix conforms to N species and M rxns
    if np.shape(nmat)[0] != nspec*2 or np.shape(nmat)[1] != nrxn:
        print("ERROR ssa.py: full stochiometry matrix does not match number of species and rxns")
        return trajectory, 1

    #create propensity array
    propensity = np.zeros((nrxn, 1))
    continuesimulation = True
    #hard code max number of iterations
    maxcounter = 100000
    counter = 0
    while continuesimulation:

        #calculate propensities
        #loop over reactions
        for j in range(nrxn):
            #init rxn propensity to zero
            propensity[j, 0] = 0
            #check for any reactant species
            if(any(nmat[:, j] > 0)):
                #accumulate product of populations - init to 1
                propensity[j, 0] = 1
                #loop over reactant species
                for i in range(nspec):
                    #loop over positive coeficients for factorial
                    if nmat[i, j] > 0:
                        for k in range(nmat[i, j]):
                            propensity[j, 0] *= (state[i, 0]-k)/(nmat[i, j]-k)
        #multiply population terms by reaction rates to get propensity
        propensity *= kvec

        #calculate propensity_sum
        propensity_sum = np.sum(propensity)

        #stop simulation if no more reactions can occour
        if propensity_sum <= 0:
            print("ssa propensity sum is zero - exiting simulation")
            continuesimulation = False #default value
            ierr = 1 #default value

        else:
            #continue simulation
            continuesimulation = True

            #choose two uniform random numbers on unit interval
            xi1, xi2 = np.random.random(2)

            #calculate exponential time interval based on propensity
            tau = np.log(1/xi1)/propensity_sum

            #choose reaction
            jrxn = np.squeeze(np.where(np.cumsum(propensity) > xi2*propensity_sum)).flat[0]

            #update populations
            state = state + dstate[:, jrxn].reshape((nspec, 1))

            #update time
            time = time + tau
            #stop simulation if runtime is complete
            if time >= runtime:
                #print("time=runtime")
                continuesimulation = False
                ierr = 0

            #append state to trajectory
            trajectory = np.hstack((trajectory, np.vstack((state, time))))

        #increment counter
        counter += 1
        #stop simulation if max number of iterations reached
        if counter >= maxcounter:
            continuesimulation = False
            print("max number of ssa iterations reached.")
            #indicate warning with ierr
            ierr = 1

    return trajectory, ierr

def plottrajectory(trajectory):
    """ utility for visualizing trajectory returned by SSA
    """
    #import matplotlib
    #matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    #import numpy as np

    #def abline(slope, intercept):
    #    """Plot a line from slope and intercept"""
    #    axes = plt.gca()
    #    x_vals = np.array(axes.get_xlim())
    #    y_vals = intercept + slope * x_vals
    #    plt.plot(x_vals, y_vals, '--')

    #plt.scatter(*zip(*trajectory))
    #abline(1, 0)
    for i in range(trajectory.shape[0]-1):
        plt.plot(trajectory[-1, :], trajectory[i, :], label=f"Species {i}")
    plt.legend()
    plt.show()

def plotweightedvalue(trajectory, weight):
    """ utility for visualizing weighted value of trajectory returned by SSA
    """
    #import matplotlib
    #matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np

    #convert weight to numpy arrays
    weight = np.array(weight)
    nspec = np.size(weight)
    weight = np.reshape(weight, (nspec, 1))

    if isinstance(trajectory, (list, )):
        #for i in range(len(trajectory)):
        #    plt.plot(trajectory[i][-1, :], weight.T.dot(trajectory[i][0:nspec, :]).T, label=f"trajectory {i}")
        for i, traj in enumerate(trajectory):
            plt.plot(traj[-1, :], weight.T.dot(traj[0:nspec, :]).T,
                     label=f"trajectory {i}")
    else:
        plt.plot(trajectory[-1, :], weight.T.dot(trajectory[0:nspec, :]).T,
                 label="Weighted Value")

    plt.legend()
    plt.show()
