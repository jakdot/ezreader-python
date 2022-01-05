"""
Examples of how to use E-Z reader.
"""

import simpy

import ezreader as ez

# words to be inputted into E-Z reader simulations are 5-tuples with the following information:
# token frequency predictability integration_time integration_failure
word1 = ez.Word('john', 5e06, 0.1, 200, 0.9)
word2 = ez.Word('sleeps', 1e05, 0.1, 25, 0.01)
word3 = ez.Word('very', 1e05, 0.1, 25, 0.01)
word4 = ez.Word('long', 1e05, 0.1, 25, 0.01)

# we create the instance of simulation
sim = ez.Simulation(sentence=[word1, word2, word3, word4], realtime=False)

# we run simulation per sim.step; each step is one action in the simulation (e.g., L1, L2, saccade programming)
while True:
    try:
        # we run a step in simulation and print what action it was, what is the time stamp (in ms).
        sim.step()
        print("Current fixation point: ", sim.fixation_point)
    except simpy.core.EmptySchedule:
        # if there are no remaining steps in the simulation we break.
        break

