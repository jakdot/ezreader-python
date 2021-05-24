"""
Examples of how to use E-Z reader.
"""

import simpy

import simulation as ez

sim = ez.Simulation(sentence=[ez.Word('john', 5e06, 0.1, 200, 0.9), ez.Word('sleeps', 1e05, 0.1, 25, 0.01), ez.Word('very', 1e05, 0.1, 25, 0.01), ez.Word('long', 1e05, 0.1, 25, 0.01)], realtime=False)

while True:
    try:
        sim.step()
        print("Current fixation point: ", sim.fixation_point)
    except simpy.core.EmptySchedule:
        break

