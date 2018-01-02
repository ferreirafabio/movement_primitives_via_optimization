import numpy as np
from scipy.optimize import minimize
from movement_primitives_optimization.helpers import math


def optimize(traj_d, start, goal, norm):
  '''
  this function minimizes the Lagrangian as specified in "Movement Primitives via Optimization" (Dragan
  2015) in equation 2 in order to adapt a demonstrated trajectory to two new endpoints (start and goal). The adaptation
  assumes as a norm a finite difference matrix of a spring damper system (new positions are calculated based on
  accelerations).
  :param traj: (T, n)
  :param start: (n,)
  :param goal: (n,)
  :param norm: (T,T) --> assert positive definite
  :return: adapted trajectory (T,n)
  '''
  assert math.is_pos_def(norm), "norm is not positive definite"

  fun = lambda traj: ((traj_d - traj).T.dot(norm)).dot(traj_d - traj)
  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - start},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - goal})


  return minimize(fun, x0=traj_d, method='SLSQP', bounds=None, constraints=cons,
   tol=1e-17, options={'ftol': 1e-17, 'disp': True, 'maxiter': 20000})


def adapt_all_dimensions(traj_d, start, goal):
  """
  This function adapts a demonstrated trajectory to a given start and goal endpoint as specified in "Movement
  Primitives via Optimization" (Dragan 2015). The adaptation assumes as a norm a finite difference matrix of a
  spring damper system (new positions are calculated based on accelerations).
  :param traj_d: (ndarray) The trajectory of shape: (length/time steps of trajectory, dimensions)
  :param start: (ndarray) 1st equality constraint, the new start point, shape: (dimensions,)
  :param goal: (ndarray) 2nd equality constraint the new goal point, shape: (dimensions,)

  :return: The adapted trajectory as an ndarray of shape [n_dim, len_traj], with n_dim being the number of
  trajectories and len_traj being the length or number of time steps of a trajectory. The order of the dimensions is
  the same as the order in the input.
  """

  K = math.get_finite_diff_matrix(traj_d.shape[0])
  M = np.transpose(K).dot(K)

  dimensions = traj_d.shape[1]
  new_trajectories = []

  for dim in range(dimensions):
    new_traj = optimize(traj_d[:, dim], start[dim], goal[dim], M)
    new_trajectories.append(new_traj.x)


  return np.asarray(new_trajectories)




