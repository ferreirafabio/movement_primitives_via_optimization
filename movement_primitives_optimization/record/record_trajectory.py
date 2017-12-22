"""Matlab code in this package provided by You Zhou (Karlsruhe Instite of Technology)"""
import matlab.engine
import numpy as np


def record_single_trajectory():
  eng = matlab.engine.start_matlab()
  eng.cd("record_n_trajectories/", nargout=0)
  ret = eng.eval('recordTrajectory', nargout=1)
  while eng.isvalid(ret):
    pass
  trajectory = np.asarray(eng.workspace['trajectory'])
  eng.quit()
  return trajectory



def record_n_trajectories(n=1):
  eng = matlab.engine.start_matlab()
  eng.cd("record/", nargout=0)
  trajectories = []
  for i in range(n):
    ret = eng.eval('recordTrajectory', nargout=1)
    while eng.isvalid(ret):
      pass
    trajectory = np.asarray(eng.workspace['trajectory'])
    trajectories.append(trajectory)

  eng.quit()

  # TODO: prune trajectories to equal length if number_trajectories > 1
  return np.array(trajectories)