"""Define AF network model Class"""

import numpy as np
from crpm.ssa import ssa
#from crpm.ffn_bodyplan import read_bodyplan
#from crpm.ffn_bodyplan import init_ffn
#from crpm.ffn_bodyplan import copy_ffn
#from crpm.fwdprop import fwdprop
#from crpm.lossfunctions import loss
#from crpm.gradientdecent import gradientdecent
#from crpm.contrastivedivergence import contrastivedivergence

class AFnetEnvironment:
    """ AF network Emulator """

    def __init__(self, patients=1, typeratio=0):
        """init population with 1000 units of metabolite X
            800 units of metabolite Y
            N=8 chemical species A, B, C, D, E, F, X, and Y
            interact by M=12 chemical reactions
            expressed as follows
            -----------
            rxn1: C -> D
            rxn2: B -> E
            rxn3: A -> B
            rxn4: A -> C
            rxn5: B -> D
            rxn6: C -> E
            rxn7: D -> F
            rxn8: E -> F
            rxn9: X -> A
            rxn10: A -> X
            rxn11: F -> Y
            rxn12: Y -> F
            -----------
            Species X and Y are held constant
            """
        #Model Parameters
        #define number of chemical species
        self.nspec = 8
        #clamp species population (1/0 = true/false): {A, B, C, D, E, F, X, Y}
        self.__clamp = [0, 0, 0, 0, 0, 0 , 1, 1]
        #init population condition
        self.ipop = [0, 0, 0, 0, 0, 0, 1000, 500]
        #simulation population scale
        self.scale = .001
        #time to equilibrate healthy patient from init population condition
        self.eqtime = 10
        #time to remove memory from previous patient state
        self.burntime = 1

        #rate vectors (M)    C->D B->E A->B A->C B->D C->E D->F E->F X->A A->X F->Y Y->F
        self.__kvec_type1 = [1.0, 0.90, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.__kvec_type2 = [0.90, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        #Full stoichiometry matrix (2N x M)
        self.__nmat = [[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0], #A reagents
                       [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #B
                       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #C
                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #D
                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #E
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #F
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #X
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #Y
                       #1  2  3  4  5  6  7  8  9  0  11 12
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #A products
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #B
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #C
                       [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #D
                       [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #E
                       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1], #F
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #X
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] #Y

        #if less than 1 patient throw warning and set patients to 1
        if patients < 1: #throw warning and set patients to 1
            print("Number of patients must be positive integer.")
            print("Will recover by setting number of patients to default value.")
            patients = 1

        #ensure type 1:2 ratio is non negative
        if typeratio < 0:
            typeratio = 0
        #set the type ratio
        self.typeratio = typeratio

        #initialize equilibrated patient state for the two subtypes
        self.newpatient(reinit=True, type=1)
        self.newpatient(reinit=True, type=2)

        #init at least one new patient: 1st patient is always of type 2 (ref type)
        self.state = self.newpatient(type=2)
        #this patient is automatically in the reference group.
        self.group = np.array([True])

        #enroll more patients if needed
        if patients > 1:
            self.enroll(patients=(patients-1))


    def newpatient(self, reinit=False, type=2):
        """returns state of new patient.
            will reinit eq state if flag is set."""

        #default runtime = burn time
        runtime = self.burntime

        #set initial populations and eq time if reinitializing
        if reinit:
            ipop = self.ipop
            runtime = self.eqtime
        else:
            #set initial pop based on type
            ipop = self.eqpop2 #default
            if type==1:
                ipop = self.eqpop1

        #set type rate vector
        kvec = self.__kvec_type2 #default
        if type==1:
            kvec = self.__kvec_type1

        #simulate new patient trajectory
        trajectory, _ = ssa(state=ipop, nmat=self.__nmat,
                            kvec=kvec, runtime=runtime,
                            clamp=self.__clamp)

        #Update Equilibrium populations
        if type == 1:
            self.eqpop1 =  np.copy(trajectory[0:self.nspec, -1].reshape((self.nspec, 1)))
        else:
            self.eqpop2 = np.copy(trajectory[0:self.nspec, -1].reshape((self.nspec, 1)))

        #scale trajectory populations
        trajectory[0:self.nspec,:] *= self.scale

        #initialize time index
        trajectory[-1, -1] = 0
        return trajectory[:, -1].reshape((self.nspec+1, 1))

    def enroll(self, patients=1):
        """enroll new patients."""
        istart = self.state.shape[1]
        #for every new patient, calculate patient state
        for ipat in range(istart, (istart+patients)):
            print("enrolling patient "+str(ipat))

            #assign patient type 2 if ratio is less than the typeratio
            assignment = False
            if np.mean(self.group) < self.typeratio:
                assignment = True
            self.group = np.append(self.group, assignment)

            #get type number
            type = 1
            if assignment:
                type = 2

            #add new patient to state
            newpatient_state = self.newpatient(type=type)
            self.state = np.hstack((self.state, newpatient_state))
