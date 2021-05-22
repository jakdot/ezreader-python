"""
Examples of how to use E-Z reader.
"""

import simpy

import simulation as ez

sim = ez.Simulation(sentence=[ez.Word('john', 1e05, 0.01), ez.Word('sleeps', 1e06, 0.8), ez.Word('long', 1e05, 0.8)], realtime=False)
while True:
    try:
        sim.step()
        print("Current fixation point: ", sim.fixation_point)
    except simpy.core.EmptySchedule:
        break

