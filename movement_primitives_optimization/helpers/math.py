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


def get_2nd_order_finite_diff_matrix(size):
  '''
  2nd order finite differencing matrix according to a spring damper system with which new positions are calculated based on
  the accelerations in a system.
  :param size: size of the quadratic matrix
  :return: the differencing matrix of shape (size, size)
   '''
  return 2 * np.diag(np.ones([size])) + np.diag(-np.ones([size - 1]), k=1) + np.diag(-np.ones([size - 1]), k=-1)


def project_norm_pos_def(M):
  """
  Projects a matrix M (norm) onto the cone of pos. (semi) def. matrices
  :param M: a square matrix - numpy array of shape (m,m)
  :return: P, the projection of M on the cone pos. semi-def. matrices
  """
  eigval, eigvec = np.linalg.eigh(M)
  eigval_pos = np.maximum(eigval, 0)
  P = eigvec.dot(np.diag(eigval_pos)).dot(eigvec.T)
  assert P.shape == M.shape
  return P
