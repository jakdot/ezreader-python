"""
Simulation of E-Z reader.
"""

# TODO:
# add noise (gamma, normal)

from collections import namedtuple
import math
import numpy as np

from numpy.random import uniform, normal
import simpy

import ezreader.utilities as ut

OPTIMAL_SACCADE_LENGTH = 7

Word = namedtuple('Word', 'token frequency predictability integration_time integration_failure')
Action = namedtuple('Action', 'name details time')

class Simulation(object):
    """
    E-Z reader simulation.
    """

    model_parameters = {
            "alpha1": 104,
            "alpha2": 3.4,
            "alpha3": 39,
            "eccentricity": 1.15,
            "delta": 0.34,
            "predictability_repeated_attention": 0.9,
            "saccade_programming": 125,
            "saccade_finishing": 25,
            "time_attention_shift": 25,
            "omega1": 6,
            "omega2": 3,
            "eta1": 0.5,
            "eta2": 0.15,
            "lambda": 0.16,
            "probability_correct_regression": 0.6 # see Reichle et al. 2009, p. 13 - last word 0.6
            }

    def __init__(self, sentence, realtime=False, noise=False, initial_time=0, initial_fixation=1, trace=True):
        """
        :param sentence: a list of Word triples representing the sentence.
        :param realtime: should simulation run in real time?
        :param noise: should noise be switched on/off?
        :param initial_time: at which simulation time does the simulation start?
        """

        if realtime:
            self.env = simpy.RealtimeEnvironment(initial_time=initial_time)
        else:
            self.env = simpy.Environment(initial_time=initial_time)

        self.env.process(self.__visual_processing__(sentence))
        self.fixation_point = initial_fixation #the point at which fixation starts (default = 1 = the first letter)
        self.attended_word = None #what word is currently attended?
        self.fixated_word = None #what word is currently fixated?
        self.last_action = None # what was the last action?
        self.trace = trace # should we print trace?
        self.__canbeinterrupted = True
        self.__plan_sacade = False
        self.__saccade = None
        self.__repeated_attention = 0 # time on repeated attention due to integration failure
        self.__fixation_launch_site = 0
        self.__word__position_dict = {}

        position = 1
        for word in sentence:
            self.__word__position_dict[(position, position+1+len(word.token))] = word.token
            position += 1+len(word.token)

        for position in self.__word__position_dict:
            if initial_fixation >= position[0] and initial_fixation <= position[1]:
                self.fixated_word = self.__word__position_dict[position]
                break

    @property
    def time(self):
        """
        Time in simulation in ms.
        """
        return 1000*self.env.now

    def __timeout__(self, time_in_ms):
        """
        Translate from ms to s and create a timeout event.
        :param time_in_ms: time (in ms)
        """
        return self.env.timeout(time_in_ms/1000)

    def __collect_action__(self, action):
        """
        Collect action and print if trace parameter of the model set to True.
        :param action: namedtuple Action
        """
        self.last_action = action

        if self.trace:
            print(self.last_action)

    def __prepare_saccade__(self, new_fixation_point, word, canbeinterrupted=True):
        """
        Prepare saccade. This function checks if a saccade can be interrupted, interrupts it if possible and sends a request to start a new saccade.
        """
        if self.__canbeinterrupted:

            # try to interrupt unless __saccade is None (at start) or RunTimeError (it was already terminated by some other process)
            try:
                self.__saccade.interrupt()
            except (AttributeError, RuntimeError):
                pass
            if (float(self.fixation_point) <  float(new_fixation_point) - len(word)/2) or (float(self.fixation_point) >  float(new_fixation_point) + len(word)/2): # start a saccade only if fixation away from the current word
                self.__saccade = self.env.process(self.__saccadic_programming__(new_fixation_point=new_fixation_point, word=word, canbeinterrupted=canbeinterrupted))

        # mark that the next saccade should be started if saccade is going on but cannot be interrupted
        else:
            if self.fixation_point != new_fixation_point:
                self.__plan_sacade = (new_fixation_point, word, canbeinterrupted)

    def __saccadic_programming__(self, new_fixation_point, word, regression=False, canbeinterrupted=True):
        """
        Generator simulating saccadic programming.

        :param new_fixation_point: where to move (in number of letters)
        :param word: what word is saccade into
        """
        self.__collect_action__(Action('Started saccade', " ".join(['Planned saccade:', str(self.fixation_point), '->',  str(new_fixation_point), 'Word:', word]), self.time))

        # labile saccade programming M1 (unless canbeinterrupted is specified as False)
        self.__canbeinterrupted = canbeinterrupted
        tM1 = self.model_parameters['saccade_programming'] #tM1, see p. 5

        try:
            # try to run the full M1 process
            yield self.__timeout__(tM1)

        except simpy.Interrupt:
            # unless it was interrupted; in that case, stop
            self.__collect_action__(Action('Interrupted saccade programming', " ".join(['Planned saccade:', str(self.fixation_point), '->',  str(new_fixation_point), 'Word:', word]), self.time))
            self.__canbeinterrupted = True

        else:

            # if the process was not interrupted, proceed to M2 (non-labile process)
            self.__canbeinterrupted = False
            self.__collect_action__(Action('Saccade programming finished', " ".join(['Planned saccade:', str(self.fixation_point), '->',  str(new_fixation_point), 'Word:', word]), self.time))

            tM2 = self.model_parameters['saccade_finishing'] #tM2

            yield self.__timeout__(tM2)

            self.__collect_action__(Action('Saccade finished', " ".join(['Planned saccade:', str(self.fixation_point), '->',  str(new_fixation_point), 'Word:', word]), self.time))
            

            intended_saccade_length = abs(self.fixation_point - new_fixation_point)

            systematic_error = (OPTIMAL_SACCADE_LENGTH - intended_saccade_length) * ( (self.model_parameters["omega1"] - math.log(self.time - self.__fixation_launch_site)) / (self.model_parameters["omega2"]))

            self.__fixation_launch_site = self.time

            self.fixation_point = normal( new_fixation_point + systematic_error, self.model_parameters["eta1"] + self.model_parameters["eta2"]*intended_saccade_length)
            
            # store what word is now fixated
            for position in self.__word__position_dict:
                if self.fixation_point >= position[0] and self.fixation_point <= position[1]:
                    self.fixated_word = self.__word__position_dict[position]
                    break

            # finally, if there was meanwhile request for another saccade (by __plan_saccade), start executing it now; otherwise set __saccade at done (None, the starting point)
            if self.__plan_sacade:
                self.__saccade = self.env.process(self.__saccadic_programming__(new_fixation_point=self.__plan_sacade[0], word=self.__plan_sacade[1], canbeinterrupted=self.__plan_sacade[2]))
                self.__plan_sacade = False
            else:

                # now two situations: either refixation, or done;
                random_draw = uniform()
                if self.model_parameters["lambda"] * abs(self.fixation_point - new_fixation_point) >= random_draw:
                    self.__saccade = self.env.process(self.__saccadic_programming__(new_fixation_point=new_fixation_point, word=word, canbeinterrupted=canbeinterrupted))
                else:
                    self.__saccade = None
                    self.__canbeinterrupted = True

    def __integration__(self, last_letter, new_fixation_point, new_fixation_point2, elem, elem_for_attention, next_elem):
        """
        Generator simulating integration.

        :param first_letter: first_letter of the word to which attention will be directed
        :param new_fixation_point: where to move in case of regression
        :param new_fixation_point2: where to move in case of done regression and moving forward
        :param elem: what element is being integrated (Word)
        :param elem_for_attention: what element will be attended to when failure
        :param next_elem: what element the attention will jump onto after success in reintegration
        """
        self.__collect_action__(Action('Started integration', " ".join(["Word:", str(elem.token)]), self.time))

        yield self.__timeout__(float(elem.integration_time))
        
        random_draw = uniform()

        # two options - either failed integration or successful
        if float(elem.integration_failure) >= random_draw:
        
            self.__collect_action__(Action('Failed integration', " ".join(["Word:", str(elem.token)]), self.time))

            # if failed integration, start saccade back to that word and attend the word again
            self.__prepare_saccade__(new_fixation_point, str(elem_for_attention.token), canbeinterrupted=False)
            self.env.process(self.__attend_again__(last_letter, new_fixation_point2, elem=elem_for_attention, next_elem=next_elem))

        else:

            self.__collect_action__(Action('Successful integration', " ".join(["Word:", str(elem.token)]), self.time))



    def __attend_again__(self, last_letter, new_fixation_point, elem, next_elem):
        """
        Attend the non-integrated word again.
        """
        old_attended_word = str(elem.token)
        if self.attended_word != old_attended_word:
            time_attention_shift = self.model_parameters["time_attention_shift"]

            yield self.__timeout__(time_attention_shift)
            self.attended_word = elem
            
            self.__collect_action__(Action('Attention shift', " ".join(["To word:", str(elem.token)]), self.time))

        # check first if you attend outside of the text (zeroth word)
        if elem.token == "None":
            time_familiarity_check = 50 # just some small number for familiarity if we jump out of text
            self.__repeated_attention += time_familiarity_check
            yield self.__timeout__(time_familiarity_check)

            self.__collect_action__(Action('L1 FAKE', " ".join(["Word:", str(self.attended_word.token)]), self.time))

            self.__prepare_saccade__(new_fixation_point, str(next_elem.token))
                
        # otherwise proceed with the standard attention
        else:
            
            distance = last_letter - self.fixation_point
            
            random_draw = uniform()

            # calculate L1, either 0 or time according to the formula in ut
            # probably should be zero when you reattend, since the word is familiar already?
            # not clear - based on Reichle et al. 2009 it seems that L1 should be done again but it might be they assume a high cloze probability the second time, so L1 effectively becomes zero; this also matches simulations from Staub (2011)

            # we assume that the second time around, predictability is close to 1 (per model parameter), hence L1 skipped:

            if self.model_parameters["predictability_repeated_attention"] > random_draw:
                time_familiarity_check = 0

            else:
                time_familiarity_check = ut.time_familiarity_check(distance=distance, wordlength=len(elem.token), frequency=elem.frequency, predictability=elem.predictability, eccentricity=self.model_parameters['eccentricity'], alpha1=self.model_parameters['alpha1'], alpha2=self.model_parameters['alpha2'], alpha3=self.model_parameters['alpha3'])
            
            self.__repeated_attention += time_familiarity_check
            
            yield self.__timeout__(time_familiarity_check)

            self.__collect_action__(Action('L1', " ".join(["Word:", str(self.attended_word.token)]), self.time))
                
            self.__prepare_saccade__(new_fixation_point, str(next_elem.token))

            # calculate L2, time according to the formula in ut
            time_lexical_access = ut.time_lexical_access(frequency=elem.frequency, predictability=elem.predictability, delta=self.model_parameters['delta'], alpha1=self.model_parameters['alpha1'], alpha2=self.model_parameters['alpha2'], alpha3=self.model_parameters['alpha3'])
                
            self.__repeated_attention += time_lexical_access

            yield self.__timeout__(time_lexical_access)

            self.__collect_action__(Action('L2', " ".join(["Word:", str(self.attended_word.token)]), self.time))
        
            self.__repeated_attention += float(elem.integration_time)
        
            yield self.__timeout__(float(elem.integration_time))
            
            self.__collect_action__(Action('Successful integration', " ".join(["Word:", str(elem.token)]), self.time))
            
            # reset attended word to continue in normal way
            self.attended_word = old_attended_word



    def __visual_processing__(self, sentence):
        """
        Generator simulating visual processing.
        """
        first_letter = 1

        for i, elem in enumerate(sentence):
            self.attended_word = elem
            # calculate distance from the current fixation to the first letter of the word
            distance = first_letter - self.fixation_point
            
            # calculate L1, either 0 or time according to the formula in ut
            random_draw = uniform()

            if float(elem.predictability) > random_draw:
                time_familiarity_check = 0

            else:
                time_familiarity_check = ut.time_familiarity_check(distance=distance, wordlength=len(elem.token), frequency=elem.frequency, predictability=float(elem.predictability), eccentricity=self.model_parameters['eccentricity'], alpha1=self.model_parameters['alpha1'], alpha2=self.model_parameters['alpha2'], alpha3=self.model_parameters['alpha3'])

            yield self.__timeout__(time_familiarity_check)
            
            yield self.__timeout__(self.__repeated_attention) # add extra time if there is a repeated processing of some previous word
            
            self.__repeated_attention = 0

            self.__collect_action__(Action('L1', " ".join(["Word:", str(elem.token)]), self.time))

            #start programming movement to the next word

            try:
                # if there is a next word, store that info
                next_elem = sentence[i+1]

            except IndexError:
                pass

            else:
                new_fixation_point = first_letter + len(elem.token) + 0.5 + len(next_elem.token)/2 # move to the middle of the next word

                self.__prepare_saccade__(new_fixation_point, str(next_elem.token))

            # calculate L2, time according to the formula in ut
            time_lexical_access = ut.time_lexical_access(frequency=elem.frequency, predictability=elem.predictability, delta=self.model_parameters['delta'], alpha1=self.model_parameters['alpha1'], alpha2=self.model_parameters['alpha2'], alpha3=self.model_parameters['alpha3'])

            yield self.__timeout__(time_lexical_access)
            
            yield self.__timeout__(self.__repeated_attention) # add extra time if there is a repeated processing of some previous word
            
            self.__repeated_attention = 0

            self.__collect_action__(Action('L2', " ".join(["Word:", str(elem.token)]), self.time))
            ########################
            #  start integration   #
            ########################

            if i > 0:
                # if there is a previous word, store that info, needed for integration
                prev_pos = first_letter - 0.5 - len(sentence[i-1].token)/2
                prev_word = sentence[i-1].token
            else:
                prev_pos = 0
                prev_elem = Word('None', 1e06, 1, 0, 0)

            random_draw = uniform()

            # this checks whether, in case of failure, you will regress to the actual word or one word before that (simplifying assumption about regressions)
            if float(self.model_parameters["probability_correct_regression"]) >= random_draw:
                self.env.process(self.__integration__(last_letter=first_letter+len(elem.token), new_fixation_point=first_letter - 0.5 + len(elem.token)/2, new_fixation_point2=new_fixation_point, elem=elem, elem_for_attention=elem, next_elem=next_elem))
            else:
                self.env.process(self.__integration__(last_letter=first_letter - 2,new_fixation_point=prev_pos, new_fixation_point2=new_fixation_point, elem=elem, elem_for_attention=prev_elem, next_elem=next_elem))

            ########################
            #   end integration    #
            ########################

            time_attention_shift = self.model_parameters["time_attention_shift"]

            yield self.__timeout__(time_attention_shift)
            
            yield self.__timeout__(self.__repeated_attention) # add extra time if there is a repeated processing of some previous word

            self.__repeated_attention = 0

            self.__collect_action__(Action('Attention shift', " ".join(["From word:", str(elem.token)]), self.time))

            first_letter += len(elem.token) + 1 #set the first letter of the new word (assuming 1 space btwn words)

    def step(self):
        """
        Make one step through simulation.
        """
        self.env.step()

    def run(self, until):
        """
        Run simulation.
        """
        self.env.run(until=until)

if __name__ == "__main__":
    #examples how to run simulation
    sleeps_fixated = []
    highest_fixation = 0
    for _ in range(1):
        sim = Simulation(sentence=[Word('john', 5e06, 0.01, 25, 0.01), Word('sleeps', 2e05, 0.01, 25, 0.01), Word('extremely', 1e03, 0.01, 25, 0.01), Word('long', 1e05, 0.01, 25, 0.01)], realtime=False, trace=False)
        #sim.run(2) #if you want to run the whole simulation
        sleeps_fixated.append(0)
        while True:
            try:
                sim.step()
                if sim.fixated_word and sim.fixation_point >= highest_fixation and sim.fixated_word != sleeps_fixated[-1]:
                    sleeps_fixated.append(sim.fixated_word)
            except simpy.core.EmptySchedule:
                break
            highest_fixation = max(highest_fixation, sim.fixation_point)

        print(sleeps_fixated)

