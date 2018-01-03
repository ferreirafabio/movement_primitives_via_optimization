import numpy as np
from scipy.linalg import polar

def is_pos_def(M):
  """ checks whether x^T * M * x > 0, M being the matrix to be checked
  :param M: the matrix to be checked
  :return: True if positive definite, False otherwise
  """
  return np.all(np.linalg.eigvals(M) > 0)


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


def get_finite_diff_matrix(size):
  '''
  finite differencing matrix according to a spring damper system with which new positions are calculated based on
  the accelerations in a system.
  :param size: size of the quadratic matrix
  :return: the differencing matrix of shape (size, size)
   '''
  return 2 * np.diag(np.ones([size])) + np.diag(-np.ones([size - 1]), k=1) + np.diag(-np.ones([size - 1]), k=-1)


def project_norm_pos_def(M):
  """
  Projects a matrix (norm) onto the space of pos. (semi) def. matrices by using polar decomposition of the form M = U P
  where U is a unitary matrix and P is a pos. semi-def. matrix. Currently, we assume that using a pos.-semi-def.
  suffices as projection onto the pos. def. space due to the numerical approach.
  :param norm: a square matrix of the form M = U P where U = unitary matrix and P a pos. semi-def. matrix
  :return: P, the pos. semi-def. matrix of the equation M = U P
  """
  U, P = polar(M)
  return P
