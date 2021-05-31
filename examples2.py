"""
Examples of how to use E-Z reader.

Simulation based on Staub (2011).
"""

import simpy
import numpy as np

import simulation as ez

def simulation(sentence):

        sim = ez.Simulation(sentence=sentence, realtime=False, initial_fixation=3.5, trace=False)

        start_time = 0

        while True:
            try:
                sim.step()
            except simpy.core.EmptySchedule:
                break
            if float(sim.fixation_point) >= 8 and start_time == 0:
                start_time = sim.time
            if float(sim.fixation_point) >= 14 or float(sim.fixation_point) < 7 and start_time != 0:
                return sim.time - start_time
        return np.nan

simulation_values = [(25, 0.01), (150, 0.01), (25, 0.6), (150, 0.6)]
#simulation_values = [(150, 0.01)]

simulation_dict_walked = {x: [] for x in simulation_values}
simulation_dict_ambled = {x: [] for x in simulation_values}

for _ in range(1000):

    for inttime, intfailure in simulation_values:
        simulation_value = simulation([ez.Word('walked', 159, 0, float(inttime), float(intfailure)), ez.Word('across', 5e03, 0, 25, 0), ez.Word('the', 1e05, 1, 25, 0), ez.Word('quad', 10, 1, 25, 0)])
        if simulation_value != 0:
            simulation_dict_walked[(inttime, intfailure)].append(simulation_value)
        #print(simulation_value)
        #input()

    for inttime, intfailure in simulation_values:
        simulation_value = simulation([ez.Word('ambled', 1, 0, float(inttime), float(intfailure)), ez.Word('across', 5e03, 0, 25, 0), ez.Word('the', 1e05, 1, 25, 0), ez.Word('quad', 10, 1, 25, 0)])
        if simulation_value != 0:
            simulation_dict_ambled[(inttime, intfailure)].append(simulation_value)
        #print(simulation_value)
        #input()


print("Walked")

for key in simulation_dict_walked:
    print(key)
    print(np.mean(simulation_dict_walked[key]))

print("Ambled")

for key in simulation_dict_ambled:
    print(key)
    print(np.mean(simulation_dict_ambled[key]))
