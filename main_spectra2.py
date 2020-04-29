""" Generate 2 group spectra of K modes """
import numpy as np
import matplotlib.pyplot as plt

def spectra2():
    """ Return data set of 600 spectra.

    Spectra consist of 20 cosine modes with uniform random frequencies
    from .1 to 10 in arbitray frequency units. Amplitudes are sampled
    from normal distribution centered on integer group number with standard
    deviation of 0.2.

    """
    #constants
    nmode = 20 #number of modes
    ngroup = 2 # number of groups
    nobv = 600 # number of observations
    minfreq = .01 # slowest mode frequency
    maxfreq = .5 # fastest mode frequency
    sigma = .2 # amplitude standard deviation
    ngrid = 1000 #spectrum grid space

    #assign random mode frequencies per group
    omega = np.random.uniform(low=minfreq, high=maxfreq, size=(nmode,ngroup))

    #assign random groups
    group = np.random.randint(ngroup, size=(1,nobv))

    #assign random mode amplitutes
    eta = np.random.normal(loc=0, scale=sigma, size=(nmode, nobv))

    #visualize distribution of ampl
    #count, bins, ignored = plt.hist(eta, 30, density=True)
    #plt.show()

    #init spectra
    spec = np.zeros((ngrid,nobv))

    #loop over groups
    for g in range(ngroup):
        ingroup = np.where(group[0,:]==g)[0]
        #construct spectra on space grid
        for i in range(ngrid):
            spec[i,ingroup] = np.sum(np.multiply(eta[:,ingroup], np.cos(omega[:,g:g+1]*i)), axis=0)

    #add group labels to first row
    spec = np.vstack((group,spec))

    #visualize spectra
    #cols = np.where(spec[0,:],"r", "b")
    #for s in range(10):
    #    plt.plot(spec[1:,s], c=cols[s])
    #plt.show()

    #visualize spectral density
    #count, bins, ignored = plt.hist(spec, 30, density=True)
    #plt.show()

    # save spectra
    np.savez("crpm/data/spectra2",spec)

if __name__ == "__main__":
    spectra2()
