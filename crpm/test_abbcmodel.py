"""Test ABBC model"""
import numpy as np
from math import ceil
from crpm.abbc_model import *
from crpm.dynamics import computeforces

#set random Seed
np.random.seed(1199167)

#define short and long term constants
short_term = 0.5
mid_term = 1.5
long_term = 3.0


def rtest_predictoutcome():
    """test accuracy of predicted outcome for health patients is constant"""

    #define cohort size
    npatients = 2

    #init healthy patients
    simulator = AbbcEnvironment(patients=npatients)

    #simulate healthy patients for long term in short term increments
    nstep = int(long_term/short_term)

    #define action taken : -1 means patients will be simulated as healthy
    action = np.repeat(-1, npatients)

    #init episode list
    episode = [simulator.state]

    #main simulation loop to generate episodes
    for step in range(nstep):
        episode += simulator.take_action(action=action, simtime=short_term)

    #episode length is 1+2*nstep consisting of intit state (5xnpat) followed by
    # next state and reward (1xnpat) repeating each time step.
    #print(episode)
    #print(len(episode))

    #---semi gradient temporal difference (0) algorithm ---
    #init hyperparameters
    alpha = .1 #learning rate
    #init Value function model
    agent = AbbcAgent(discount=1.0)
    #loop over episodes
    for patient in range(npatients):
        #state = [nstep]
        #state += episode[0][:,patient] #get inital state
        state = np.append(episode[0][:,patient],nstep).reshape((6,1)) #get inital state

        print(state)
        #loop over time steps in episode
        for k in range(1,nstep+1):
            #get next state and reward
            #nextstate = [nstep-k]
            #nextstate = episode[k*2-1][:,patient]
            nextstate = np.append(episode[k*2-1][:,patient],nstep-k).reshape((6,1))

            reward = episode[k*2][patient]

            #get magnitude for forces
            magnitude = alpha * (reward + agent.discount * agent.get_value(nextstate)
                                 - agent.get_value(state))
            #compute forces
            forces = computeforces(agent.prednet, state, 0, "iden")

            #update model
            for layer in forces:
                index = layer["layer"]
                agent.prednet[index]["weight"] +=  magnitude * layer["fweight"]
                agent.prednet[index]["bias"] += magnitude * layer["fbias"]

        state = np.copy(nextstate)


    #make predictions
    state = np.append(episode[0][:,patient],nstep).reshape((6,1)) #get inital state
    print(agent.get_value(state))

    #Value function approximates outcome return at time horizon.
    assert(False)

    ##define action taken
    #action = np.repeat(2, npatients)
    ##main simulation loop
    #for step in range(nstep):
    #    _, drugy_reward[step,:] = simulator.take_action(action=action, simtime=short_term)




def rtest_power_and_reward_crossover():
    """test sample size increases as effect size between drugx and drugy crossover with time.
    Samplesize is calculated assuming 80% power and 0.05 significance.
    Assert max samplesize is greater than 100 patients.
    Assert DrugX has greater effect than DrugY at short term while DrugY has
    greater effect at long term.
    """
    #define pilot cohort size
    npatients = 10

    #simulate drugX and drugY policies for long term in short term increments
    nstep = int(long_term/short_term)

    #init rewards per patient per time for drugx and drugy policies
    drugx_reward = np.zeros((nstep,npatients))
    drugy_reward = np.zeros((nstep,npatients))

    #init drugX simulation
    simulator = AbbcEnvironment(patients=npatients)
    #define action taken
    action = np.repeat(1, npatients)
    #main simulation loop
    for step in range(nstep):
        _, drugx_reward[step,:] = simulator.take_action(action=action, simtime=short_term)

    #init drugY simulation
    simulator = AbbcEnvironment(patients=npatients)
    #define action taken
    action = np.repeat(2, npatients)
    #main simulation loop
    for step in range(nstep):
        _, drugy_reward[step,:] = simulator.take_action(action=action, simtime=short_term)

    #calculate sample size required to resolve effect size for each simulation step.
    zalpha = 1.96 #critical zvalue for p=1-alpha/2
    zbeta = 0.8416 #critical zvalue for p=1-beta
    zsquared = (zalpha + zbeta)**2
    #calculate the diference in effect size
    delta = (np.mean(drugx_reward, axis=1) - np.mean(drugy_reward, axis=1))
    samplesize = np.divide((np.var(drugx_reward, axis=1) + np.var(drugx_reward, axis=1))*zsquared,delta**2)
    print(samplesize)
    print(delta)

    #assert max sample size is greater than 100
    assert np.max(samplesize)>100

    #assert max sample size at is at least 50 times the mins at short and long terms
    assert np.max(samplesize)>50*samplesize[0]
    assert np.max(samplesize)>50*samplesize[-1]

    #assert DrugX is better short term than DrugY
    assert delta[0]>0
    #assert DrugY is better in long term than DrugX
    assert delta[-1]<0
