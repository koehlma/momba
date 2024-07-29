import random


import racetrack

from momba import engine

scenario = racetrack.model.Scenario(
    racetrack.tracks.BARTO_SMALL,
    start_cell=None,
    underground=racetrack.model.Underground.ICE,
    random_start=True,
)
network = racetrack.model.construct_model(scenario)

explorer = engine.Explorer.new_discrete_time(network)

(state,) = explorer.initial_states

transition = next(iter(state.transitions))

print(transition.instances)


random.seed(0)

print(transition.destinations.pick().state.global_env)
