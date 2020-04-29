"""Test deep Q learning on abbc model"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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


def test_td0_vfa():
    """test value function approximator has resonable prospective and
    retrospective accuracy.
    Value function is return at horizon time.
    """
    #constants
    maxstep = 500#0 #max q learning steps
    stepsize = .5 #time step increment
    icohort = 20#5 #initial cohort size at trial start
    enrate = 20#5 #enrollment rate per visit
    engoal = 200 #enrollment goal
    vfrac = 0.2 #fraction of patients in validation set
    diagt = .2 #time patients live with undiagnosed disease
    gamma = 1.0 #discount factor
    explprob = 0.0 #exploration rate (no exploration)
    expldecay = 1.0 #exploration rate decay (constant exploration rate)

    #conduct deep q learning
    agent = QAgent(discount=gamma,
                   exploration_rate=explprob,
                   exploration_rate_decay=expldecay)
    simulator = AbbcEnvironment(patients=icohort, t=diagt, validation_fraction=vfrac)

    rewardlog, actionlog, obv, pred, sigma = run_simulation(agent, simulator,
                                                            maxstep,
                                                            stepsize=stepsize,
                                                            enroll_rate=enrate,
                                                            enroll_goal=engoal,
                                                            minibatchsize=10)
    print("final rewards")
    print(rewardlog[-1,:])
    print("stdev rewards")
    print(sigma)

    print("initial outcome")
    print(simulator.outcome(simulator.istate))
    print("final outcomes")
    print(simulator.outcome(simulator.state))

    print("final outcomes 95% confidence interval")
    ci = np.exp(np.vstack((rewardlog[-1,:]-2*sigma[-1],rewardlog[-1,:]+2*sigma[-1])))
    print((simulator.outcome(simulator.istate)*ci).T)

    #plot outcome
    fig = plt.figure(1, figsize=(7, 2.5))
    plt.plot(range(6,6+obv.shape[0]),obv[:,0], 'k')
    #plot predictions
    ub = pred[:,0]+2*sigma
    lb = pred[:,0]-2*sigma
    plot_mean_and_CI(pred[:,0], ub, lb, color_mean='b', color_shading='b')

    assert False

    #bring back validation patients
    patients = np.where(simulator.validation)[0]
    simulator.state[-1,patients] = 0 #reset time
    simulator.istate[:,patients] = simulator.state[:,patients] #reset initial state
    simulator.withdrawn[patients] = False #these patients are no longer withdrawn
    simulator.visit[patients] = 0 #reset visit counter
    #continue simulation without updating model to measure prognsostic strength
    rewardlog, actionlog, obv, pred, sigma = run_simulation(agent, simulator,
                                                            maxstep=10,
                                                            stepsize=stepsize,
                                                            update=False,
                                                            enroll_rate=0,
                                                            enroll_goal=engoal)

    print("final rewards")
    print(rewardlog[-1,:])
    print("stdev rewards")
    print(sigma)


    print("initial outcome")
    print(simulator.outcome(simulator.istate))
    print("final outcomes")
    print(simulator.outcome(simulator.state))

    print("final outcomes 95% confidence interval")
    ci = np.exp(np.vstack((rewardlog[-1,:]-2*sigma[-1],rewardlog[-1,:]+2*sigma[-1])))
    print((simulator.outcome(simulator.istate)*ci).T)


    assert False


def test_benchmark_policies_for_short_and_long_term_rewards():
    """test a drug2 policy has higher long term rewards than the drug2 policy
        and drug1 policy has higher short term rewards

        We will use a chort of 10 patients to decide if rewards are different based
        on previous power analysis.
    """
    from scipy import stats
    #constants
    maxstep = 6 #max simulation step
    cohort_size = 5

    #benchmark simulation with drug1 agent
    agent = Drug1Policy()
    simulator = AbbcEnvironment(patients=cohort_size)
    rewardlog, actionlog = run_simulation(agent, simulator, maxstep, update=False)
    drug1_short_reward = rewardlog[0, :] #np.sum(rewardlog, axis=0)
    drug1_long_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("drug1 rewards")
    print(rewardlog)
    print(actionlog)
    #assert all actions were drug 1
    #assert(all(action == 1 for action in actionlog))
    assert(np.all(actionlog == 1 ))

    #benchmark simulation with drug2 agent
    agent = Drug2Policy()
    simulator = AbbcEnvironment(patients=cohort_size)
    rewardlog, actionlog = run_simulation(agent, simulator, maxstep, update=False)
    drug2_short_reward = rewardlog[0, :] #np.sum(rewardlog, axis=0)
    drug2_long_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("drug2 rewards")
    print(rewardlog)
    print(actionlog)
    #assert all actions were drug 2
    assert(np.all(actionlog == 2 ))
    #assert(all(action == 2 for action in actionlog))

    #assert drug2 rewards are better in long run on average
    assert drug2_long_reward.mean() > drug1_long_reward.mean()

    #assert long rewards are significantly different
    _, pvalue = stats.ttest_ind(drug1_long_reward, drug2_long_reward)
    assert pvalue < .05

    #assert drug1 rewards are better in short run on average
    assert drug1_short_reward.mean() > drug2_short_reward.mean()

    #assert short rewards are significantly different
    _, pvalue = stats.ttest_ind(drug1_short_reward, drug2_short_reward)
    assert pvalue < .05


def test_agent_randomwalk():
    """test dql agent will take random drug with no exploration decay rate.
    """
    #constants
    nstep = 100 #max q learning steps
    factor = .001 #reward and state scaling factor

    #conduct deep q learning
    agent = QAgent(discount=0.95, exploration_rate=1.0, exploration_rate_decay=1.0)
    simulator = AbbcEnvironment()
    #store current state
    state = np.copy(simulator.state)
    state *= factor

    #get withdrawn npatients
    withdrawn = np.copy(simulator.withdrawn)
    #init action statisitcs
    action = np.zeros((nstep,1))
    #ask for next action many times to get statistics on action chosen
    for step in range(nstep):
        action[step,] = agent.get_next_action(state, withdrawn)
    print("randomw walk")
    print(action)

    #assert distribution of actions are statistically the same
    margin = 1.5/np.sqrt(nstep)
    assert np.mean(action==0) < 1/3 + margin
    assert np.mean(action==1) < 1/3 + margin
    assert np.mean(action==2) < 1/3 + margin
    assert np.mean(action==0) > 1/3 - margin
    assert np.mean(action==1) > 1/3 - margin
    assert np.mean(action==2) > 1/3 - margin

def test_agent_updates_Q_properly():
    """test dql agent tends to prefer to take any drug over no treatment.
    """
    #constants
    training_steps = 24 #max q learning steps
    training_cohort = 3
    testing_steps = 6
    testing_cohort = 10

    #conduct deep q learning
    agent = QAgent(discount=0.95, exploration_rate=1.0, exploration_rate_decay=0.9)
    simulator = AbbcEnvironment(patients=training_cohort, validation_fraction=.3)
    rewardlog, actionlog = run_simulation(agent, simulator, training_steps, enroll_rate=1)
    print("dql training")
    print(actionlog)
    print(rewardlog)

    #simulate trained dql agent with fixed policy
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    print("dql testing")
    print(actionlog)
    print(rewardlog)

    #Takes any drug more than 80% of the time
    assert np.mean(actionlog>0) > .80


def test_agent_selects_drug1():
    """test dql agent will preferentially select drug 1 with discount rate = 0.
    """
    #constants
    training_steps = 24 #max q learning steps
    training_cohort = 3
    testing_steps = 6
    testing_cohort = 10

    #conduct deep q learning
    agent = QAgent(discount=0.0, exploration_rate=1.0, exploration_rate_decay=0.9)
    simulator = AbbcEnvironment(patients=training_cohort, validation_fraction=.3)
    rewardlog, actionlog = run_simulation(agent, simulator, training_steps, enroll_rate=1)
    print("dql training")
    print(actionlog)
    print(rewardlog)

    #simulate trained dql agent with fixed policy
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    print("dql testing")
    print(actionlog)
    print(rewardlog)

    #Takes drug1 more than 50% of the time
    assert np.mean(actionlog == 1) > .50

    assert False


def test_policy_against_naive_short_term_solution():
    """test a dql policy has higher long term rewards than the drug2 policy.
    """
    #constants
    training_steps = 24 #max q learning steps
    training_cohort = 3
    testing_steps = 6   #simulation steps of 0.5 time units
    testing_cohort = 10

    #conduct deep q learning
    agent = QAgent(discount=0.95, exploration_rate=1.0, exploration_rate_decay=.90)
    simulator = AbbcEnvironment(patients=training_cohort, validation_fraction=.3)
    rewardlog, actionlog = run_simulation(agent, simulator, training_steps, enroll_rate = 1)
    print("dql training")
    print(actionlog)
    print(rewardlog)

    #simulate trained dql agent with fixed policy
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    dql_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("dql testing")
    print(actionlog)
    print(rewardlog)

    #benchmark simulation with drug1 agent (will always take drug1)
    agent = Drug1Policy()
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, drug1_actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    drug1_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("drug1 rewardlog")
    print(rewardlog)

    #assert trained dql policy rewards are better in long run than drug1 policy
    assert drug1_reward.mean() < dql_reward.mean()

    #assert two rewards are significantly different
    _, pvalue = stats.ttest_ind(drug1_reward, dql_reward)
    assert pvalue < .05
    print (pvalue)

    #benchmark simulation with drug2 agent (will always take drug2)
    agent = Drug2Policy()
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    drug2_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("drug2 rewardlog")
    print(rewardlog)

    #assert trained dql policy rewards statistically the same as drug2 policy rewards
    _, pvalue = stats.ttest_ind(drug2_reward, dql_reward)
    assert pvalue > .05
    print (pvalue)

    assert False
