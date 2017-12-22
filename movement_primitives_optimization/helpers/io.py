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