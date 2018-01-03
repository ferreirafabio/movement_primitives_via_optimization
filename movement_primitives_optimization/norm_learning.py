import numpy as np
from scipy.optimize import minimize
import pandas as pd
from movement_primitives_optimization.helpers import math



def inner_minimization_per_dimension(traj_i, traj_j, norm):
  """
  # TODO: write documentation
  :param traj_i:
  :param traj_j:
  :param norm:
  :return:
  """
  fun = lambda traj: ((traj_i-traj).T.dot(norm).dot(traj_i-traj) - math.loss_function(traj, traj_j))

  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - traj_j[0]},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - traj_j[1]})

  return minimize(fun, x0=np.ones(shape=(traj_i.shape[0])), method='SLSQP', bounds=None, constraints=cons,
                  tol=1e-17, options={'ftol': 1e-17, 'disp': True, 'maxiter': 20000})



def inner_minimization(traj_i, traj_j, norm):
  assert traj_i.shape[1] == traj_j.shape[1]

  dimensions = traj_i.shape[1]
  new_trajectories = []

  for dim in range(dimensions):
    new_traj = inner_minimization_per_dimension(traj_i[:, dim], traj_j[:, dim], norm)
    new_trajectories.append(new_traj.fun)


  return np.asarray(new_trajectories)



def learn_norm(demonstrations, init_norm, alpha, iterations=1000):
  assert demonstrations, "no trajectory given"
  assert alpha > 0
  assert math.is_pos_def(init_norm)


  if isinstance(demonstrations, pd.DataFrame):
    # flatten required to convert 2d array to 1d
    demonstrations = demonstrations.values.flatten()

  norm = grad_m = init_norm


  for i in range(iterations):
    print("iteration: " + str(i) + "\n")
    for traj_i in demonstrations:
      for traj_j in demonstrations:
        traj_ij = inner_minimization(traj_i, traj_j, norm)
        grad_m = np.sum((traj_i - traj_j).dot((traj_i - traj_j).T) - (traj_i - traj_ij).dot((traj_i - traj_ij).T))
    norm -= alpha * grad_m

  return norm
