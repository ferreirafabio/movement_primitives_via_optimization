import numpy as np
from scipy.optimize import minimize


def loss_function(traj, traj_j):
  """
  indicator loss function for trajectories
  :param traj: (ndarray) first trajectory (the one to be minimized) of shape (n,)
  :param traj_j: (ndarray) second trajectory (from the demonstrations) of shape (n,)
  :return: 0 if trajectories are of the same shape and equal in terms of their elements, 1 otherwise
  """
  if np.array_equal(traj, traj_j):
    return 0
  return 1


def inner_minimization(traj_i, traj_j, norm):
  """
  # TODO: write documentation
  :param traj_i:
  :param traj_j:
  :param norm:
  :return:
  """
  # TODO: test implementation
  fun = lambda traj: ((traj_i-traj).T.dot(norm).dot(traj_i-traj) - loss_function(traj, traj_j))

  cons = ({'type': 'eq', 'fun': lambda traj: traj[0] - traj_j[0]},
          {'type': 'eq', 'fun': lambda traj: traj[-1] - traj_j[1]})

  return minimize(fun, x0=np.ones(shape=(traj_i.shape[0])), method='SLSQP', bounds=None, constraints=cons,
                  tol=1e-17, options={'ftol': 1e-17, 'disp': True, 'maxiter': 20000})


