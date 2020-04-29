"""Define ABBC model - agent and environment"""

import numpy as np
from crpm.ssa import ssa
from crpm.ffn_bodyplan import read_bodyplan
from crpm.ffn_bodyplan import init_ffn
from crpm.ffn_bodyplan import copy_ffn
from crpm.fwdprop import fwdprop
from crpm.lossfunctions import loss
from crpm.gradientdecent import gradientdecent
from crpm.contrastivedivergence import contrastivedivergence

#from enums import *
#import random

class AbbcEnvironment:
    """ ABBC Emulator """

    def __init__(self, patients=1, group_fraction=0, dropout_rate=0,
                 t=None, sigma=0):
        """init population with 1000 units of metabolite X
            N=5 chemical species A, B, B_prime, C, and X
            interact by M=9 chemical reactions
            expressed as follows
            -----------
            rxn1: X -> A
            rxn2: A -> X
            rxn3: X -> C
            rxn4: C -> X
            rxn5: A -> C
            rxn6: A -> B
            rxn7: A -> B_prime
            rxn8: B -> C
            rxn9: B_prime -> C
            -----------
            Species X is held constant
            """
        #Model Parameters
        #define number of chemical species
        self.nspec = 5
        #clamp species population (1/0 = true/false)
        self.__clamp = [0, 0, 0, 0, 1]
        #init population condition
        self.ipop = [0, 0, 0, 0, 1000]
        #simulation population scale
        self.scale = .001
        #time to equilibrate healthy patient from init population condition
        self.eqtime = 10
        #time to remove memory from previous healthy patient state
        self.burntime = 1
        #define max number of visits enrolled patients remain under observation
        self.maxvisit = 10 #6
        #define time horizon patients remain under observation
        self.timehorizon = 5.0 #3.0
        # set undiagnosed time and variance
        self.undiagnosedtime = t
        self.undiagnosedvar = sigma

        #rate vectors (M)
        self.__kvec_healthy = [1.0, 0.5, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, 5.0]
        self.__kvec_disease = [1.0, 0.5, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 5.0]
        self.__kvec_drug1 = [1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 5.0]
        self.__kvec_drug2 = [1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 5.0]

        #Full stoichiometry matrix (2N x M)
        self.__nmat = [[0, 1, 0, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0, 0, 0, 0],

                       [1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 1, 0, 0, 1, 1],
                       [0, 1, 0, 1, 0, 0, 0, 0, 0]]


        #if less than 1 patient throw warning and set patients to 1
        if patients < 1: #throw warning and set patients to 1
            print("Number of patients must be positive integer.")
            print("Will recover by setting number of patients to default value.")
            patients = 1

        #ensure validation fraction is non negative
        if group_fraction < 0:
            group_fraction = 0
        #set the validation fraction
        self.group_fraction = group_fraction

        #ensure the dropout rate is not negative
        if dropout_rate < 0:
            dropout_rate = 0
        #set the dropout_rate
        self.dropout_rate = dropout_rate

        #create healthy patient state
        self.eqpop = self.newpatient(reinit=True)[0:self.nspec]/self.scale

        #init at least one new patient
        self.state = self.newpatient()
        #save the patient's initial state
        self.istate = np.copy(self.state)
        #this patient is automatically not in the validation group.
        self.group = np.array([False])
        #this patient is has not yet withdrawn because just enrolled
        self.withdrawn = np.array([False])
        #init this patient's visit counter
        self.visit = np.array([0])

        #enroll more patients if needed
        if patients > 1:
            self.enroll(patients=(patients-1))


    def outcome(self, states):
        """returns the outcome values for every state provided.
        """
        #define health metric as linear combination of state features
        # A is begnign
        # B is bad
        # B_prime is worse
        # C is good
        # X outcome is indepenant of X
        # time outcome is independant of time
        weight = np.array([0, -1, -2, 1, 0, 0]).reshape((1,6))
        return np.squeeze(weight.dot(states))

    def newpatient(self, reinit=False):
        """returns state of newly diagnosed patient or patient living with
        undiagnosed disease for time t. will reinit eq state flag set."""

        #default to simulate patient
        continuesim = True

        #Get a new healthy patient state
        if reinit: #Equilibrate healthy patient from scratch
            trajectory, _ = ssa(state=self.ipop, nmat=self.__nmat,
                                kvec=self.__kvec_healthy, runtime=self.eqtime,
                                clamp=self.__clamp)
            #do not simulate patient any further
            continuesim = False
        else: #start new patient from eqstate
            trajectory, _ = ssa(state=self.eqpop, nmat=self.__nmat,
                                kvec=self.__kvec_healthy, runtime=self.burntime,
                                clamp=self.__clamp)

        #default to last trajectory index for return state
        idx = -1
        dt = .5 #default time increment

        #simulate acute onset of disease until diagnosis
        while continuesim:
            if self.undiagnosedtime is not None:
                #simulate for a specific amount of time
                dt = self.undiagnosedtime + np.random.normal(0,self.undiagnosedvar)
                continuesim = False
            #simulate patient
            state = np.copy(trajectory[0:self.nspec, -1])
            trajectory, _ = ssa(state=state, nmat=self.__nmat,
                                kvec=self.__kvec_disease, runtime=dt,
                                clamp=self.__clamp)
            #calculate outcome of patient for each state along trajectory
            value = self.outcome(trajectory)
            #check for diagnosis
            if (self.undiagnosedtime is None) and np.any(value < 250):
                #Retrieve state upon diagnosis defined when health measure drops below 250
                idx = np.where(value.T < 250)[0][0]
                continuesim = False

        #scale trajectory populations
        trajectory[0:self.nspec,:] *= self.scale
        #initialize time horizon
        trajectory[-1,idx] = self.timehorizon
        return trajectory[:, idx].reshape((self.nspec+1, 1))

    def enroll(self, patients=1):
        """enroll new patients."""
        istart = self.state.shape[1]
        #for every new patient, calculate patient state
        for ipat in range(istart, (istart+patients)):
            print("enrolling patient "+str(ipat))
            #add new patient to state
            newpatient_state = self.newpatient()
            self.state = np.hstack((self.state, newpatient_state))
            #save new patient's starting state
            self.istate = np.hstack((self.istate, newpatient_state))
            #assign patient to validation group if ratio is under the group_fraction
            isval = False
            if np.mean(self.group) < self.group_fraction:
                isval = True
            self.group = np.append(self.group, isval)
            #this patient is has not yet withdrawn because just enrolled
            self.withdrawn = np.append(self.withdrawn, False)
            #init this patient's visit counter
            self.visit = np.append(self.visit, 0)

    def dropout(self):
        """keep track of patients who have withdrawn"""
        patients = self.state.shape[1]

        #track patients for no more than maxvisits
        self.withdrawn = np.where(self.visit > self.maxvisit, True, self.withdrawn)

        ##drop patients that have past the time horizon
        #self.withdrawn = np.where(self.state[-1,:] <= 0, True, self.withdrawn)

        #randomly drop patients with fixed probability
        self.withdrawn = np.where(np.random.random_sample(patients) < self.dropout_rate, True, self.withdrawn)

    def take_action(self, action, simtime=.5):
        """simulate treatment for discrete time interval"""
        patients = self.state.shape[1]
        #get current time to horizion for all patients
        time0 = self.state[-1,:]
        # get starting outcome for all patients
        outcome0 = self.outcome(self.state)
        #outcome0 = self.outcome(self.istate)
        #simulate action for each patient
        for patient in range(patients):
            kvec = None #default action is no simulation
            if action[patient] == 0:
                # No Treatment: use disease kvector
                kvec = self.__kvec_disease
            #if action[patient] == 1:
            #    # drug1 treatment
            #    kvec = self.__kvec_drug1
            #if action[patient] == 2:
            #    # drug2 treatment
            #    kvec = self.__kvec_drug2

            #conduct simulation for valid action
            if kvec is not None:
                trajectory, _ = ssa(state=self.state[0:self.nspec, patient]/self.scale,
                                    nmat=self.__nmat, kvec=kvec,
                                    runtime=simtime, clamp=self.__clamp)
                #scale trajectory populations
                trajectory[0:self.nspec,:] *= self.scale
                # subtract simulation time from starting time to horizon
                trajectory[-1, -1] = time0[patient] - trajectory[-1, -1]
                #save state at end of simulation
                self.state[:, patient] = trajectory[:, -1]
                #increment patient's visit counter
                self.visit[patient] += 1

        #Calcualte reward as log of realized outcome return
        reward = np.log(np.divide(self.outcome(self.state),outcome0))

        #return new states and rewards for all patients
        return np.copy(self.state), reward

class ObservationalPolicy:
    """ This agent will always take action 0 (observe only). """
    def __init__(self):
        """ create observational policy agent """
        pass #nothing to initialize

    def get_next_action(self, state, withdrawn):
        """ returns an array of actions with size of the number of columns in state"""
        # always do nothing: action == 0 unless patient is withdrawn
        return np.where(withdrawn, np.nan, 0)

    def update(self, state, action, reward, new_state, validation=None):
        pass # nothing to update! policy never changes!!


class Drug1Policy:
    """ This agent will always say to take drug 1. """
    def __init__(self):
        """ create drug1 policy agent """
        pass #nothing to initialize

    def get_next_action(self, state, withdrawn):
        """ returns an array of actions with size of the number of columns in state"""
        # always take drug1: action == 1 unless patient is withdrawn
        return np.where(withdrawn, np.nan, 1)
        #return np.repeat(1, state.shape[1])

    def update(self, state, action, reward, new_state, validation=None):
        pass # nothing to update! policy never changes!!

class Drug2Policy:
    """ This agent will always say to take drug 2. """
    def __init__(self):
        """ create drug2 policy agent """
        pass #nothing to initialize

    def get_next_action(self, state, withdrawn):
        """ returns an array of actions with size of the number of columns in state"""
        # always take drug2: action == 2
        return np.where(withdrawn, np.nan, 2)
        #return np.repeat(2, state.shape[1])

    def update(self, state, action, reward, new_state, validation=None):
        pass # nothing to update! policy never changes!!

class QAgent:
    def __init__(self, discount=0.95, exploration_rate=1.0, exploration_rate_decay=.99, target_every=2):
        """ define deep network hyperparameters"""
        self.discount = discount # how much future rewards are valued w.r.t. current
        self.exploration_rate = exploration_rate # initial exploration rate
        self.exploration_rate_decay = exploration_rate_decay # transition from exploration to expliotation
        self.target_every = target_every #how many iterations to skip before we swap prediciton network with target network

        #retrieve the body plan
        #input has 6 neurons, one for each metabolite conc. and one for time horizon
        #output has 1 neuron, representing the only action and its value function approximation
        #~~output has 3 neurons, representing the Q values for each of the 3 actions
        #~~   action 0 is no treatment, action 1 is drug1 Tx, and and action 2 is for drug2 Tx
        self.bodyplan = read_bodyplan("crpm/data/abbc_bodyplan.csv")

        #define prediction network
        self.prednet = init_ffn(self.bodyplan)
        self.loss = None #current prediction error

        #init the target network(s)
        self.targetnet1 = init_ffn(self.bodyplan)
        self.targetnet2 = init_ffn(self.bodyplan)
        self.targetnet3 = init_ffn(self.bodyplan)
        self.targetnet4 = init_ffn(self.bodyplan)
        #with the prediciton network
        #self.targetnet = copy_ffn(self.prednet)

        #init counter used to determine when to update target network with prediction network
        self.iteration = 0

   # Ask model to estimate value for current state (inference)
    def get_value(self, state):
        # prediction network input: array of 6 values representing metabolite conc. and time horizon
        #  output: Value of anticipated return at time horizon from current state

        prediction, _state = fwdprop(state, self.prednet)
        return prediction

   # Ask model to calcualte value keeping current policy
    def get_target_value(self, state):
        # target network input: array of 6 values representing metabolite conc. and time horizon
        #  output: Value of anticipated return at time horizon from current state
        pred1, _ = fwdprop(state, self.targetnet1)
        pred2, _ = fwdprop(state, self.targetnet2)
        pred3, _ = fwdprop(state, self.targetnet3)
        pred4, _ = fwdprop(state, self.targetnet4)
        #stack predictions
        predstack = np.stack((pred1, pred2, pred3, pred4), axis = -1)
        predmin = np.amin(predstack, axis=2)
        predmax = np.amax(predstack, axis=2)
        # Return weighted average (ala BEAR alg, Kumar, 2019)
        return .75*predmin+.25*predmax

    def get_next_action(self, state, withdrawn):
        """ returns an array of actions with size of the number of columns in state"""
        #get number of patients
        patients = state.shape[1]
        greedy_actions = self.greedy_action(state)
        random_actions = self.random_action(patients)
        actions = np.where(np.random.random_sample(patients) > self.exploration_rate, greedy_actions, random_actions)
        return np.where(withdrawn, np.nan, actions)

    # Which action has bigger Q-value, estimated by our model (inference).
    def greedy_action(self, state):
        # argmax picks the higher Q-value and returns the index for every patient
        return np.argmax(self.get_value(state), axis=0)

    def random_action(self, size):
        return np.random.randint(self.bodyplan[-1]["n"], size=size)



    def pretrain(self, state, validation=None):
        """ will pretrain deep network model by contrastive divergence """

        #make sure input all have the same number of observations
        nobv = state.shape[1]
        failcheck = False
        if validation is not None and validation.shape[0] != nobv:
            failcheck = True
        if failcheck:
            print("runtime error in pretrain: inconsistent number of observations!")
            return

        #get network input size
        nfeat = state.shape[0] #network input size

        if validation is None:
            #manually set validation data to False
            validation = np.full(state.shape[0],False)

        #partition out validation patients from dataset
        intrain = ~validation
        nobv = np.sum(intrain)
        #exit if too few participated
        if nobv < 1:
            print("too few participants found for training")
            return
        #otherwise proceed with training
        data = state[:, intrain].reshape((nfeat, nobv))

#Left off here - need to pop off last layer in model and add random weight to target and prediction nets

        #return untrained autoencoder
        _, autoencoder = contrastivedivergence(self.prednet, data, maxepoch=0)
        print(autoencoder)

        #calculate initial mean squared error
        pred, _ = fwdprop(data, autoencoder)
        icost, _ = loss("mse", pred, data)
        print(icost)


        #train model
        _, autoencoder = contrastivedivergence(self.prednet, data, maxepoch=100)

        #calculate final mean squared error
        pred, _ = fwdprop(data,autoencoder)
        cost, _ = loss("mse", pred, data)

        #print(autoencoder)
        print(icost)
        print(cost)

        #reinit the target network(s)
        #with the prediciton network
        #self.targetnet = copy_ffn(self.prednet)
        self.targetnet1 = copy_ffn(self.prednet)
        self.targetnet2 = copy_ffn(self.prednet)
        self.targetnet3 = copy_ffn(self.prednet)
        self.targetnet4 = copy_ffn(self.prednet)


    def train(self, state, action, reward, new_state, validation=None):
        """ will train deep network model by gradient decent """

        #make sure input all have the same number of observations
        nobv = state.shape[1]
        failcheck = False
        if new_state.shape[1] != nobv:
            failcheck = True
        if action.shape[0] != nobv:
            failcheck = True
        if reward.shape[0] != nobv:
            failcheck = True
        if validation is not None and validation.shape[0] != nobv:
            failcheck = True
        if failcheck:
            print("runtime error in train: inconsistent number of observations!")
            return

        # Ask the model for the Q values of the current state (inference)
        state_values = self.get_value(state)
        #print("state_values")
        #print(state_values)

        # Ask the model for the Q values of the new state (target)
        new_state_values = self.get_target_value(new_state)
        #print("new_state_values")
        #print(new_state_values)

        #get network input size
        nfeat = state.shape[0] #network input size
        nlabels = state_values.shape[0] #network output size (actions)
        #print("nfeat and nlabels(actions)")
        #print(nfeat)
        #print(nlabels)

        #print("actions")
        #print(action)

        # loop over actions
        for iact in range(nlabels):
            #get patients who took this action
            patients = np.where(action == iact, True, False)
            #update Q Values if any patients took this action
            if np.sum(patients) > 0:
                state_values[iact, patients] = (reward[patients] +
                                                self.discount *
                                                np.amax(new_state_values[:, patients]))

        # Train prediction network
        if validation is None or np.sum(validation) < 1:
            #get data from patients that participated
            intrain = np.squeeze(~np.isnan(action))
            nobv = np.sum(intrain)
            #exit if too few participated
            if nobv < 1:
                print("too few participants found for training")
                return
            #otherwise proceed with training
            traindata = state[:, intrain].reshape((nfeat, nobv))
            #print("training data")
            #print(traindata)
            #print("training labels")
            #print(state_values[:, intrain].reshape((nlabels, nobv)))
            _, self.loss, _ = gradientdecent(self.prednet,
                                             traindata,
                                             state_values[:, intrain].reshape((nlabels, nobv)),
                                             "mse", maxepoch=1, healforces=False)
        else:
            #partition out validation patients from dataset
            intrain = np.logical_and(~validation, ~np.isnan(action))
            invalid = np.logical_and(validation, ~np.isnan(action))
            nobv = np.sum(intrain)
            nobv_v = np.sum(invalid)
            #exit if too few participated
            if nobv < 1:
                print("too few participants found for training")
                return
            if nobv_v < 1:
                print("too few participants found for validation")
                return
            #otherwise proceed with training
            traindata = state[:, intrain].reshape((nfeat, nobv))
            validata = state[:, invalid].reshape((nfeat, nobv_v))
            _, self.loss, _ = gradientdecent(self.prednet,
                                             traindata,
                                             state_values[:, intrain].reshape((nlabels, nobv)),
                                             "mse",
                                             validata=validata,
                                             valitargets=state_values[:, invalid].reshape((nlabels, nobv_v)),
                                             earlystop=True,
                                             healforces=True)
        print("loss")
        print(self.loss)
        print("bias")
        print(self.prednet[-1]["bias"])
        print("weights")
        print(self.prednet[-1]["weight"])

    def update(self, state, action, reward, new_state, validation=None):

        # Train our model with new data
        self.train(state, action, reward, new_state, validation)

        # Periodically, shift the prediction network into the target network queue
        if self.iteration % self.target_every == 0:
            tempnet = copy_ffn(self.prednet)
            self.prednet = copy_ffn(self.targetnet1)
            self.targetnet1 = copy_ffn(self.targetnet2)
            self.targetnet2 = copy_ffn(self.targetnet3)
            self.targetnet3 = copy_ffn(self.targetnet4)
            self.targetnet4 = copy_ffn(tempnet)

        # Finally shift our exploration_rate toward zero (less gambling)
        self.exploration_rate *= self.exploration_rate_decay

        #increment iteration counter
        self.iteration += 1

#def main q learning loop
def run_simulation(agent, simulator, maxstep, stepsize=.5, update=True,
                   enroll_rate=0, enroll_goal=10, minibatchsize=10,
                   online=True, file="buffer"):
    """
    Define Q learning orchestration

    Args:
        agent : determines policy, at every simulation step will decide which
            action to take for every patient in the simulator.
        simulator: determines the evolution of multiple patients in a simulation
            (cohort). Simulator evolves the state of each patient for one
            simulation step given the action dermined by the agent and calculates
            the reward of that state-action pair.
        maxstep: total integer number of simulation steps
        update: Boolean flag when true will allow agent to update the policy.
            When false, the agent will apply its current policy for the duration
            of the simulation.
        enroll_rate: integer number of new patients to add to the
            simulator at every simulation step.
    ChangeLog:
        + offline parameter tells wheather to read or write previous buffer data
        + file parameter is name where replay buffer will be read/written.
        + Randomly enroll new patients - Let simulator determine dropout rate
        + Have training, validation, and tesing patients
            - testing and validation patients enrolled together at 7:3 ratio
                + validation patients used for naive early stopping (overfitting)
            - testing patients enrolled once policy is set - used for reporting
    """

    #read replay buffer for offline learning
    if not online:
        buffer = np.load(file+".npz")
        pid_buffer = buffer['pid']
        group_buffer = buffer['group']
        visit_buffer = buffer['visit']
        state_buffer = buffer['state']
        action_buffer = buffer['action']
        reward_buffer = buffer['reward']
        new_state_buffer = buffer['new_state']
        #pretrain agent
        agent.pretrain(state_buffer,group_buffer)


    #define early stopping frequency
    earlystopfreq = 20#100

    #init policy bias and variance
    bias = np.empty(13)
    sigma = np.empty(13)

    #init learning and step counter
    step = 0
    learning_error = None
    continuelearning = True
    while continuelearning:

        print("- - - - S T E P - - - -  "+str(step))

        #sample policy for online learning and create new replay buffer
        if online:
            #drop out patients who need to leave study
            simulator.dropout()

            #enroll new patients if haven't reached enrollment goal
            if enroll_rate > 0 and simulator.withdrawn.shape[0] < enroll_goal:
                simulator.enroll(enroll_rate)

            #store current state
            state = np.copy(simulator.state)

            #get withdrawn npatients
            withdrawn = np.copy(simulator.withdrawn)

            #query agent for the next action
            action = agent.get_next_action(state, withdrawn)

            #take action, get new state and reward
            new_state, reward = simulator.take_action(action, simtime=stepsize)

            #get patients with valid actions (technically should be defined by withdrawn)
            patients = np.logical_not(withdrawn)
            pid = np.where(np.logical_not(withdrawn))[0]

            #init replay buffer at first simulation step
            if step == 0:
                pid_buffer = np.copy(pid)
                group_buffer = np.copy(simulator.group[patients])
                visit_buffer = np.copy(simulator.visit[patients])
                state_buffer = np.copy(state[:,patients])
                action_buffer = np.copy(action[patients])
                reward_buffer = np.copy(reward[patients])
                new_state_buffer = np.copy(new_state[:,patients])
            elif not np.all(withdrawn):#accumulate replay buffer
                pid_buffer = np.append(pid_buffer, np.copy(pid))
                group_buffer = np.append(group_buffer, np.copy(simulator.group[patients]))
                visit_buffer = np.append(visit_buffer, np.copy(simulator.visit[patients]))
                state_buffer = np.hstack((state_buffer, np.copy(state[:,patients])))
                action_buffer = np.append(action_buffer, np.copy(action[patients]))
                reward_buffer = np.append(reward_buffer, np.copy(reward[patients]))
                new_state_buffer = np.hstack((new_state_buffer, np.copy(new_state[:,patients])))

        #let agent update policy
        if update:
            #prepare mini-batch
            #Select random training experiences
            intrain = np.where(np.logical_not(group_buffer))[0]
            minsize = min(minibatchsize,intrain.shape[0])
            patients = np.random.choice(intrain, size=minsize, replace=False)
            #perform update step
            agent.update(state_buffer[:,patients],
                         action_buffer[patients],
                         reward_buffer[patients],
                         new_state_buffer[:,patients],
                         group_buffer[patients])

            #stop learning if loss function is nan
            if np.isnan(agent.loss):
                continuelearning = False

            #naive early stopping - use validation set periodically
            if step%earlystopfreq==0:
                #Select random validation experiences
                invalid = np.where(group_buffer)[0]
                #validratio = invalid.shape[0]/intrain.shape[0]
                #minsize = min((minibatchsize*invalid.shape[0])//intrain.shape[0], invalid.shape[0])
                #minsize = min(minibatchsize, invalid.shape[0])
                if invalid.shape[0]>0:#minsize>0:
                    #valid_patients = np.random.choice(invalid, size=minsize, replace=False)
                    #calculate QAgent's current pred error using targetnet
                    #Caclulate QAgent's prediction error on validation set.
                    #create arrays to hold first and second moments of error per visit interval
                    etau1 = np.empty(13)
                    etau2 = np.empty(13)
                    #loop over vist intervals
                    for idx in range(13):
                        #init visit interval cumulative error
                        etau1[idx] = 0
                        etau2[idx] = 0
                        #init visit interval sample counter
                        msecount = 0
                        #get visit interval index
                        tau = idx-6
                        #loop over validation patients
                        #invalid = np.where(group_buffer)[0]
                        for pid in np.unique(pid_buffer[invalid]):
                            #get samples pertaining to this patient
                            psamples = np.where(pid_buffer==pid)[0]
                            #loop over visits
                            for visit in np.unique(visit_buffer[psamples]):
                                #if visit+interval index exists for this patient then
                                if np.isin(visit+tau,visit_buffer[psamples]):
                                    #get visit+interval index
                                    vidx = np.where(visit_buffer[psamples]==visit+tau)[0][0]
                                    #get observed outcome for visit+interval
                                    obstau = simulator.outcome(state_buffer[:,psamples[vidx]])
                                    #get visit index
                                    vidx = np.where(visit_buffer[psamples]==visit)[0][0]
                                    #get observed outcome for visit
                                    obs = simulator.outcome(state_buffer[:,psamples[vidx]])
                                    #construct state to prognose
                                    progstate = state_buffer[:,psamples[vidx]] #start with visit state
                                    #change time to indicate visit time interval
                                    #progstate[-1] = state_buffer[-1,vtidx]-state_buffer[-1,vidx]
                                    progstate[-1] = stepsize*tau
                                    #make sure progstate is nx1
                                    progstate = progstate.reshape((-1,1))
                                    #calculate predicted return for (state(visit),time interval)
                                    rhat = np.squeeze(agent.get_target_value(progstate))
                                    #accumulate square error for visit interval
                                    resi = rhat - np.log(obstau/obs)
                                    etau1[idx] += resi
                                    etau2[idx] += resi*resi
                                    #increment visit interval sample counter
                                    msecount += 1
                                #end if
                            #end loop over visits
                        #end loop over patients
                        #save cumulative second moment
                        #normalize visit interval accumulated error
                        etau1[idx] /= msecount
                        etau2[idx] /= msecount
                        #print("increment")
                        #print(tau)
                        #print(msecount)
                    #end loop over visit intervals
                    bias = etau1
                    sigma = np.sqrt(etau2-etau1*etau1)
                    print("bias")
                    print(etau1)
                    print("MSE")
                    print(etau2)
                    print("stdev")
                    print(sigma)

                    #define current error
                    curr_err = np.mean(etau2)
                    #curr_err = np.sum(etau2*np.exp(-np.abs(etau2-etau2.shape[0]/2)))

                    #save variance if first time
                    if learning_error is None:
                        learning_error = curr_err

                    #stop learning if current error is greater than previous error
                    if curr_err>learning_error:
                        print("early stopping!")
                        continuelearning = False

                    #update learning error
                    learning_error = curr_err

        #complete learning step
        step += 1

        #stop learning after maxsteps
        if step > maxstep:
            continuelearning = False

    #save replay buffer at end of online learning
    if online:
        np.savez(file,
                 pid=pid_buffer,
                 group=group_buffer,
                 visit=visit_buffer,
                 state=state_buffer,
                 action=action_buffer,
                 reward=reward_buffer,
                 new_state=new_state_buffer
                 )

    #Return predictions and observations on validation set.
    #init Prediction array with dimensions (ninterval,npatient)
    invalid = np.where(group_buffer)[0]
    patients = np.unique(pid_buffer[invalid])
    npats = patients.shape[0]
    pred = np.empty((13,npats))
    nvisit = np.max(visit_buffer)
    obv = np.full((nvisit,npats),np.nan)
    #loop over validation patients
    for pidx in range(npats):
        #get patient id
        pid = patients[pidx]
        #get samples pertaining to this patient
        psamples = np.where(pid_buffer==pid)[0]
        #loop over visits to get outcomes
        for visit in np.unique(visit_buffer[psamples]):
            #get visit index
            vidx = np.where(visit_buffer[psamples]==visit)[0][0]
            #get observed outcome for visit
            obv[visit-1,pidx] = simulator.outcome(state_buffer[:,psamples[vidx]])
            #make predictions based off mid visit 3 state (midpoint)
            if visit == 0:
                #construct state to prognose based on visit0 state
                progstate = state_buffer[:,psamples[vidx]]
                #loop over visit intervals
                for idx in range(13):
                    #get visit interval index
                    tau = idx-6
                    #edit time to indicate prognosis time interval
                    progstate[-1] = stepsize*tau
                    #make sure progstate is nx1
                    progstate = progstate.reshape((-1,1))
                    #calculate predicted return for (state(visit3),time interval)
                    pred[idx,pidx] = agent.get_value(progstate)
                #end loop over intervals
            #end if visit number is 0
        #end loop over visits
    #end loop over patients
    print("predictions")
    print(pred)
    print("observations")
    print(obv)


    return obv, pred, bias, sigma
