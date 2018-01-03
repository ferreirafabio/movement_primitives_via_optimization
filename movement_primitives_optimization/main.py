"""
implementation of the paper Dynamic Motion Primitives via Optimization (Dragan et. al 2015)
"""

from movement_primitives_optimization.trajectory_adaptation import *
from movement_primitives_optimization.helpers import io
from movement_primitives_optimization.record import record_trajectory
from movement_primitives_optimization.tests import two_d_examples
from movement_primitives_optimization import norm_learning


"""run example adaptation 1"""
#two_d_examples.simple_trajectory_example()
"""run example adaptation 2 (requires matlab installation) """
#two_d_examples.record_trajectories_and_adapt(number_trajectories=2)


""" record n trajectories and store them (requires matlab installation)"""
#trajectories = record_trajectory.record_n_trajectories(n=10)
#io.pickle_trajectories(trajectories, "./data", "trajectories_digit_1.pickle")

""" load trajectories """
#trajectories_df = io.load_trajectories("./data/trajectories.pickle")
#print(trajectories_df)


""" adapt loaded trajectories """
#trajectories_adapted = two_d_examples.adapt_recorded_trajectories(trajectories_df, plot=False)
#print(trajectories_adapted)

""" MMP (equations 19 and 20) """
user_demonstrations = io.load_trajectories("./data/trajectories_digit_1.pickle")
user_demonstrations = user_demonstrations.values.flatten()
length_shortest_traj = io.get_shortest_trajectory_length(user_demonstrations)
print(length_shortest_traj)
user_demonstrations = io.trim_trajectories(user_demonstrations, length_shortest_traj)

init_norm = math.get_finite_diff_matrix(size=length_shortest_traj)
learned_norm = norm_learning.learn_norm(demonstrations=user_demonstrations, init_norm=init_norm, iterations=10)
print(learned_norm)
# TODO: test learning norm
# TODO: test if pos. semi-def. P of pole decomposition is sufficient for projection (which assumes a pos. def. matrix)


