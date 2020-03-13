"""Test deep Q learning on abbc model"""
import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats
from crpm.abbc_model import *

#set random Seed
np.random.seed(1199167)

#import matplotlib.patches as mpatches
#from matplotlib.colors import colorConverter as cc
#import numpy as np

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    """ utility for plotting lines with error bars
    """
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)


def run_trial():
    """analyze learned policy's prospective and
    retrospective accuracy.
    Value function is return at horizon time.
    """
    #constants
    maxstep = 500#00#500 #max q learning steps
    stepsize = .5 #time step increment
    icohort = 1200 #initial cohort size at trial start
    enrate = 0#50 #enrollment rate per visit
    engoal = 1200 #enrollment goal
    vfrac = 1/3.0 #fraction of patients in validation set
    mut = .2 #average time patients live with undiagnosed disease
    sigmat = .06 #stdev in time patients live with undiagnosed disease
    gamma = 1.0 #discount factor
    explprob = 0.0 #exploration rate (no exploration)
    expldecay = 1.0 #exploration rate decay (constant exploration rate)
    target_every = 1 #frequency to swap target and prediction networks
    minibatchsize = 1#32 # size of training minibatch
    bufferfile = "buffer" #name of file containing replay buffer for offline learning
    regendata = False # logical to regenerate replay buffer


    if(regendata):
        #reset ABBC environment for observational study
        simulator = AbbcEnvironment(patients=icohort,
                                    t=mut, sigma=sigmat,
                                    group_fraction=vfrac)
        #use observation only policy
        agent = ObservationalPolicy()
        #generate data for later offline learning
        _, _, _, _ = run_simulation(agent, simulator, maxstep=12, stepsize=stepsize,
                                    update=False,
                                    enroll_rate=enrate, enroll_goal=engoal,
                                    online=True, file=bufferfile)

    #conduct offline off-policy learning to optimize interventional policy
    simulator = AbbcEnvironment()
    agent = QAgent(discount=gamma, target_every=target_every)
    obv, pred, bias, sigma = run_simulation(agent, simulator, maxstep,
                                            stepsize=stepsize,
                                            online=False, file=bufferfile,
                                            minibatchsize=minibatchsize)
    print("reward bias")
    print(bias)
    print("reward stdev")
    print(sigma)

    #print("initial outcome")
    #print(simulator.outcome(simulator.istate))
    #print("final outcomes")
    #print(simulator.outcome(simulator.state))

    #plot outcomes for first 2 validataion patients
    fig = plt.figure(1, figsize=(7, 4.5))

    #choose first patient
    pid = 0
    plt.plot(range(6,6+obv.shape[0]),obv[:,pid], 'k')
    #plot predictions
    pbar = obv[0,pid]*np.exp(pred[:,pid]-bias)
    ub = obv[0,pid]*np.exp(pred[:,pid]-bias+2*sigma)
    lb = obv[0,pid]*np.exp(pred[:,pid]-bias-2*sigma)
    plot_mean_and_CI(pbar, ub, lb, color_mean='b', color_shading='b')

    #choose second patient
    pid = 1
    plt.plot(range(6,6+obv.shape[0]),obv[:,pid], 'k')
    #plot predictions
    pbar = obv[0,pid]*np.exp(pred[:,pid]-bias)
    ub = obv[0,pid]*np.exp(pred[:,pid]-bias+2*sigma)
    lb = obv[0,pid]*np.exp(pred[:,pid]-bias-2*sigma)
    plot_mean_and_CI(pbar, ub, lb, color_mean='g', color_shading='g')

    #render plot
    plt.show()

    assert False
if __name__ == "__main__":
    run_trial()
