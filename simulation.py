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
        self.__canbeinterrupted = True
        self.__plan_sacade = False
        self.__saccade = None
        self.trace = trace

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

    def __saccadic_programming__(self, new_fixation_point, word):
        """
        Generator simulating saccadic programming.
        """
        self.last_action = Action('Started saccade', " ".join(['Planned saccade:', str(self.fixation_point), '->',  str(new_fixation_point), 'Word:', word]), self.time)

        if self.trace:
            print(self.last_action)

        self.__canbeinterrupted = True
        tM1 = self.model_parameters['saccade_programming'] #tM1, see p. 5

        try:
            # try to run the full process
            yield self.__timeout__(tM1)

        except simpy.Interrupt:
            # unless it was interrupted; in that case, stop
            self.last_action = Action('Interrupted saccade programming', " ".join(['Planned saccade:', str(self.fixation_point), '->',  str(new_fixation_point), 'Word:', word]), self.time)
            if self.trace:
                print(self.last_action)

        else:
            self.__canbeinterrupted = False
            self.last_action = Action('Saccade programming finished', " ".join(['Planned saccade:', str(self.fixation_point), '->',  str(new_fixation_point), 'Word:', word]), self.time)

            if self.trace:
                print(self.last_action)

            tM2 = self.model_parameters['saccade_finishing'] #tM2

            yield self.__timeout__(tM2)

            self.last_action = Action('Saccade finished', " ".join(['Planned saccade:', str(self.fixation_point), '->',  str(new_fixation_point), 'Word:', word]), self.time)

            if self.trace:
                print(self.last_action)

            self.fixation_point = new_fixation_point

            if self.__plan_sacade:
                self.__saccade = self.env.process(self.__saccadic_programming__(new_fixation_point=self.__plan_sacade[0], word=self.__plan_sacade[1]))
                self.__plan_sacade = False

    def __integration__(self, elem):
        """
        Generator simulating integration.
        """
        self.last_action = Action('Started integration', " ".join(["Word:", str(elem.token)]), self.time)

        if self.trace:
            print(self.last_action)

        integration_failure = 0.3 # only for testing right now
        
        yield self.__timeout__(30) #30 ms - just a random example
        
        self.last_action = Action('Finished integration', " ".join(["Word:", str(elem.token)]), self.time)

        if self.trace:
            print(self.last_action)

        if integration_failure > 0.4:

                if self.__canbeinterrupted:

                    # try to interrupt unless __saccade is None (at start) or RunTimeError (it  was already terminated by some other process)
                    try:
                        self.__saccade.interrupt()
                    except (AttributeError, RuntimeError):
                        pass

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

            yield self.__timeout__(time_familiarity_check)

            self.last_action = Action('L1', " ".join(["Word:", str(self.attended_word.token)]), self.time)

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

                # interrupt if the saccade is being programmed to the word whose L1 has just been finished
                if self.__canbeinterrupted:

                    # try to interrupt unless __saccade is None (at start) or RunTimeError (it  was already terminated by some other process)
                    try:
                        self.__saccade.interrupt()
                    except (AttributeError, RuntimeError):
                        pass
                    self.__saccade = self.env.process(self.__saccadic_programming__(new_fixation_point=new_fixation_point, word=str(next_elem.token)))

                # mark that the next saccade should be started if saccade is going on but cannot be interrupted
                elif self.__saccade and not self.__canbeinterrupted:
                    self.__plan_sacade = (new_fixation_point, str(next_elem.token))

            time_lexical_access = ut.time_lexical_access(frequency=elem.frequency, predictability=elem.predictability, delta=self.model_parameters['delta'], alpha1=self.model_parameters['alpha1'], alpha2=self.model_parameters['alpha2'], alpha3=self.model_parameters['alpha3'])

            yield self.__timeout__(time_lexical_access)

            self.last_action = Action('L2', " ".join(["Word:", str(self.attended_word.token)]), self.time)

            if self.trace:
                print(self.last_action)
            
            self.env.process(self.__integration__(elem=elem))

            time_attention_shift = self.model_parameters["time_attention_shift"]

            yield self.__timeout__(time_attention_shift)

            self.last_action = Action('Attention shift', " ".join(["From word:", str(self.attended_word.token)]), self.time)

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
    #examples how to run simulation
    sim = Simulation(sentence=[Word('john', 5e06, 0.1), Word('sleeps', 2e05, 0.8), Word('long', 1e05, 0.8)], realtime=False)
    #sim.run(2) #if you want to run the whole simulation
    while True:
        try:
            sim.step()
            print("Current fixation point: ", sim.fixation_point)
        except simpy.core.EmptySchedule:
            break

