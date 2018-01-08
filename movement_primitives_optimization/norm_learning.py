import numpy as np
from scipy.optimize import minimize
import pandas as pd
from movement_primitives_optimization.helpers import math


def inner_minimization_per_dimension(traj_i, traj_j, norm):
  """
  Applies Lagrangian optimization (SLSQP method) for the trajectories provided.
  :param traj_i: First trajectory of shape (time steps of trajectory, )
  :param traj_j: Second trajectory of shape (time steps of trajectory, )
  :param norm: A norm under which the optimization process is executed.
  :return: A scipy OptimizeResult object
  """
  fun = lambda traj: ((traj_i-traj).T.dot(norm).dot(traj_i-traj) - math.loss_function(traj, traj_j))

  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - traj_j[0]},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - traj_j[1]})

  return minimize(fun, x0=np.ones(shape=(traj_i.shape[0])), method='SLSQP', bounds=None, constraints=cons,
                  tol=1e-17, options={'ftol': 1e-17, 'disp': True, 'maxiter': 20000, "disp": False})


def inner_minimization(traj_i, traj_j, norm):
  """
  Applies the right term of eq. 19 in "Movement Primitives via Optimization" (Dragan et al., 2015) via Lagrangian
  optimization (SLSQP method with constraints as specified in the paper). Each dimension is optimized separately and
  their min-values compose a new vector of shape (# dimensions,).
  :param traj_i: First trajectory of shape (time steps of trajectory, dimensions)
  :param traj_j: Second trajectory of shape (time steps of trajectory, dimensions)
  :param norm: A norm under which the optimization process is executed.
  :return: A vector of shape (# dimensions,) that is composed of the min-values of each separate dimension
  Lagrangian-optimization
  """
  assert traj_i.shape[1] == traj_j.shape[1]

  dimensions = traj_i.shape[1]
  new_trajectories = []

  for dim in range(dimensions):
    new_traj = inner_minimization_per_dimension(traj_i[:, dim], traj_j[:, dim], norm)
    new_trajectories.append(new_traj.fun)


  return np.asarray(new_trajectories)


def learn_norm(demonstrations, init_norm, alpha=0.95, iterations=1000):
  """
  Implementation of norm learning from the paper "Movement Primitives via Optimization" (Dragan et al., 2015)
  Specifically, this function learns a norm given that the user provides not only demonstrations but also adaptations
  by applying Maximum Margin Planning. The function iteratively applies the following three steps,
  given pairs of trajectories (traj_i, traj_j) \in DxD (D being the set of user demonstrations):
    1) compute the optimal solution to the "inner minimization problem" (right term in eq. 19)
    2) compute the gradient update for the norm with a hyper-parameter alpha, update the norm
    3) project the updated norm to the space of pos. def. matrices, repeat
  :param demonstrations: the trajectories, can be a pandas DataFrame or a list of ndarrays with shape (time steps,
  dimensions)
  :param init_norm: the initial norm from where we the norm updates start from
  :param alpha: learning rate for the norm update
  :param iterations: number of iterations the norm should be updates
  :return: the learned norm of the same shape as init_norm
  """
  assert demonstrations, "no trajectory given"
  assert alpha > 0
  assert math.is_pos_def(init_norm)


  if isinstance(demonstrations, pd.DataFrame):
    # flatten required to convert 2d array to 1d
    demonstrations = demonstrations.values.flatten()

  norm = init_norm


  for i in range(iterations):
    print("iteration: " + str(i+1) + "\n")
    grad_m = np.zeros(norm.shape)
    for traj_i in demonstrations:
      for traj_j in demonstrations:
        traj_ij = inner_minimization(traj_i, traj_j, norm)
        grad_m += (traj_i - traj_j).dot((traj_i - traj_j).T) - (traj_i - traj_ij).dot((traj_i - traj_ij).T)
    norm -= alpha * grad_m
    # according to eq 20, project M onto the space of pos. def. matrices by polar decomposition
    norm = math.project_norm_pos_def(norm)
  return norm
