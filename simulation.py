"""
Simulation of E-Z reader.
"""

from collections import namedtuple

import simpy

import utilities as ut

Word = namedtuple('Word', 'token frequency predictability')
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
            "saccade_programming": 125,
            "saccade_finishing": 25,
            "time_attention_shift": 25
            }

    def __init__(self, sentence, realtime=False, noise=False, initial_time=0, trace=True):
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
        self.fixation_point = 1 #the point at which fixation starts (default = 1 = the first letter)
        self.attended_word = None
        self.last_action = None
        self.__canbeinterrupted = False
        self.__plan_sacade = False
        self.__saccade = None
        self.trace = trace

    @property
    def time(self):
        """
        Time in simulation in ms.
        """
        return 1000*self.env.now

    def __saccadic_programming__(self, new_fixation_point, word):
        """
        Generator simulating saccadic programming.
        """
        self.last_action = Action('Started saccade', 'Fixation point: '+str(new_fixation_point) + " Word: " + word, self.time)
        if self.trace:
            print(self.last_action)
        self.__canbeinterrupted = True
        tM1 = self.model_parameters['saccade_programming']
                #tM1, see p. 5
        try:
            # try to run the full process
            yield self.env.timeout(tM1/1000)
        except simpy.Interrupt:
            # unless it was interrupted; in that case, stop
            self.last_action = Action('Interrupted saccade programming', 'Fixation point: '+str(new_fixation_point) + " Word: " + word, self.time)
            if self.trace:
                print(self.last_action)
        else:
            self.__canbeinterrupted = False
            self.last_action = Action('Saccade programming', 'Fixation point: '+str(new_fixation_point) + " Word: " + word, self.time)
            if self.trace:
                print(self.last_action)
            tM2 = self.model_parameters['saccade_finishing']
            yield self.env.timeout(tM2/1000)
            self.last_action = Action('Saccade finishing', 'Fixation point: '+str(new_fixation_point) + " Word: " + word, self.time)
            if self.trace:
                print(self.last_action)
            self.fixation_point = new_fixation_point
            if self.__plan_sacade:
                self.__saccade = self.env.process(self.__saccadic_programming__(new_fixation_point=self.__plan_sacade[0], word=self.__plan_sacade[1]))
                self.__plan_sacade = False


    def __visual_processing__(self, sentence):
        """
        Generator simulating visual processing.
        """
        first_letter = 1
        for i, elem in enumerate(sentence):
            self.attended_word = elem
            # calculate distance from the current fixation to the first letter of the word
            distance = first_letter - self.fixation_point
            time_familiarity_check = ut.time_familiarity_check(distance=distance, wordlength=len(elem.token), frequency=elem.frequency, predictability=elem.predictability, eccentricity=self.model_parameters['eccentricity'], alpha1=self.model_parameters['alpha1'], alpha2=self.model_parameters['alpha2'], alpha3=self.model_parameters['alpha3'])
            yield self.env.timeout(time_familiarity_check/1000)
            self.last_action = Action('L1', 'Word: '+self.attended_word.token, self.time)
            if self.trace:
                print(self.last_action)
            try:
                # if there is a next word, store that info
                next_elem = sentence[i+1]
            except IndexError:
                pass
            else:
                #start programming movement to the next word
                new_fixation_point = first_letter + len(elem.token) + 0.5 + len(next_elem.token)/2 # move to the middle of the next word

                # interrupt if the saccade is being programmed to the word whose L1 is done
                if self.__saccade and self.__canbeinterrupted:
                    self.__saccade.interrupt()
                    self.__saccade = self.env.process(self.__saccadic_programming__(new_fixation_point=new_fixation_point, word=str(next_elem.token)))

                # mark that the next saccade should be started if saccade is going on but cannot be interrupted
                elif self.__saccade and not self.__canbeinterrupted:
                    self.__plan_sacade = (new_fixation_point, str(next_elem.token))

                # else, no movement is currently going on, start a new saccade programming
                else:
                    self.__saccade = self.env.process(self.__saccadic_programming__(new_fixation_point=new_fixation_point, word=str(next_elem.token)))

            time_lexical_access = ut.time_lexical_access(frequency=elem.frequency, predictability=elem.predictability, delta=self.model_parameters['delta'], alpha1=self.model_parameters['alpha1'], alpha2=self.model_parameters['alpha2'], alpha3=self.model_parameters['alpha3'])
            yield self.env.timeout(time_lexical_access/1000)
            self.last_action = Action('L2', 'Word: '+self.attended_word.token, self.time)
            if self.trace:
                print(self.last_action)
            time_attention_shift = self.model_parameters["time_attention_shift"]
            yield self.env.timeout(time_attention_shift/1000)
            self.last_action = Action('Attention shift', 'From word: '+self.attended_word.token, self.time)
            if self.trace:
                print(self.last_action)
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
    sim = Simulation(sentence=[Word('john', 1e05, 0.01), Word('sleeps', 2e05, 0.8), Word('long', 1e05, 0.8)], realtime=False)
    #sim.run(2)
    while True:
        sim.step()
        print(sim.fixation_point)
        input()


