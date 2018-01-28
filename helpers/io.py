import pandas as pd
import os


def pickle_trajectories(trajectories, destination_path, filename):
  """

  :param trajectories: an ndarray of shape (n_traj,) with n_traj being the number of overall trajectories
  :param destination_path: the absolute path to the output directory
  :return: the full path to the pickle file
  """

  # create a dict from the trajectories
  trajectories_df = pd.DataFrame(trajectories)

  assert os.path.exists(destination_path), "invalid path to output directory"
  full_path = os.path.join(destination_path, filename)
  trajectories_df.to_pickle(full_path)
  print("Dumped trajectories dict pickle to ", full_path)
  return full_path


def load_trajectories(path_to_pickle):
  """

  :param path_to_pickle:
  :return:
  """

  assert os.path.isfile(path_to_pickle), "invalid path to output directory"
  df = pd.read_pickle(path_to_pickle)
  return df


def get_shortest_trajectory_length(trajectories):
  """
  determines and returns the shortest sequence length of trajectories given by a list of ndarrays in which each
  element represents a single (possibly multi-dimensional) trajectory.


  :param trajectories: an ndarray of shape [#number of trajectories] containing trajectory as lists of shape [
  timesteps, dimension]
  :return: the shortest sequence of trajectories
  """
  return min([trajectories[i].shape[0] for i in range(trajectories.shape[0])])


def trim_trajectories(trajectories, trim_length):
  """
  trims all provided trajectories according to trim_length in its sequence length.
  :param trajectories: an ndarray of shape [#number of trajectories] containing trajectory as lists of shape [
  timesteps, dimension]
  :param trim_length: maximum length the trajectories should have after processing
  :return: resulting shape of the trajectories will be [:trim_length, dimension] if trim_length > timesteps
  """
  return [trajectories[i][:trim_length] for i in range(trajectories.shape[0])]


