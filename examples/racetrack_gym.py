import racetrack

from momba import gym

scenario = racetrack.model.Scenario(
    racetrack.tracks.BARTO_SMALL,
    start_cell=None,
    underground=racetrack.model.Underground.ICE,
)
network = racetrack.model.construct_model(scenario)

env = gym.create_generic_env(
    network,
    next(
        instance for instance in network.instances if instance.automaton.name == "car"
    ),
    "goalProbability",
)

env.reset()

print(env.available_transitions)
