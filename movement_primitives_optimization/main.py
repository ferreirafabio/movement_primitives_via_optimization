from movement_primitives_optimization.trajectory_adaptation import *
from movement_primitives_optimization.helpers import io
from movement_primitives_optimization.record import record_trajectory
from movement_primitives_optimization.tests import two_d_examples


"""run example adaptation 1"""
#two_d_examples.simple_trajectory_example()
"""run example adaptation 2 (requires matlab installation) """
#two_d_examples.record_trajectories_and_adapt(number_trajectories=2)


""" record n trajectories and store them (requires matlab installation)"""
#trajectories = record_trajectory.record_n_trajectories(n=5)
#io.pickle_trajectories(trajectories, "./data", "trajectories.pickle")

""" load trajectories """
trajectories_df = io.load_trajectories("./data/trajectories.pickle")
print(trajectories_df)


""" adapt loaded trajectories """
two_d_examples.adapt_recorded_trajectories(trajectories_df)

